/**
 * Fails a model request whose provider goes silent. The run's abort signal
 * only covers cancellation, and the liveness heartbeat keeps a waiting run
 * looking healthy, so without this a gateway that accepts the connection and
 * never answers (LiteLLM holding a request in its own queue) pins the run
 * "in progress" and its queue slot indefinitely.
 *
 * Only provider silence is timed: the body timer runs while a read is pending,
 * so a consumer that stops pulling (tools executing) never trips it. That is
 * why this lives on fetch rather than in `streamText`'s `timeout`, whose
 * chunk and step timers keep running through tool execution.
 */

const DEFAULT_STALL_TIMEOUT_MS = 120_000;

type StallPhase = "headers" | "body";

export class ProviderStallError extends Error {
  /**
   * Set only before any response: undici's code, which the AI SDK retries as a
   * network error. A mid-stream stall carries none, because the SDK would wrap
   * it as "Failed to process successful response" and lose this message, which
   * the board's transient-failure check matches to re-dispatch the card.
   */
  readonly code?: "UND_ERR_HEADERS_TIMEOUT";

  constructor(phase: StallPhase, timeoutMs: number) {
    const after =
      timeoutMs >= 1000 ? `${Math.round(timeoutMs / 1000)}s` : `${timeoutMs}ms`;
    super(
      phase === "headers"
        ? `LLM provider stalled: no response within ${after}`
        : `LLM provider stalled: stream sent no data for ${after}`,
    );
    this.name = "ProviderStallError";
    if (phase === "headers") this.code = "UND_ERR_HEADERS_TIMEOUT";
  }
}

/** `0` disables the timeout; anything unparseable falls back to the default. */
export function stallTimeoutMsFromEnv(value: string | undefined): number {
  const trimmed = value?.trim();
  if (!trimmed || !/^\d+$/.test(trimmed)) return DEFAULT_STALL_TIMEOUT_MS;
  return Number(trimmed);
}

function withDeadline<T>(
  pending: Promise<T>,
  timeoutMs: number,
  onTimeout: () => Error,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(onTimeout()), timeoutMs);
  });
  return Promise.race([pending, deadline]).finally(() => clearTimeout(timer));
}

type BaseFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export function createStallTimeoutFetch(
  timeoutMs: number,
  // Resolved per call so a fetch swapped in later (tests) is honored.
  baseFetch: BaseFetch = (input, init) => globalThis.fetch(input, init),
): typeof fetch {
  const stallFetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const stalled = new AbortController();
    const signal = init?.signal
      ? AbortSignal.any([init.signal, stalled.signal])
      : stalled.signal;
    const stall = (phase: StallPhase) => {
      const error = new ProviderStallError(phase, timeoutMs);
      stalled.abort(error);
      return error;
    };

    const response = await withDeadline(
      baseFetch(input, { ...init, signal }),
      timeoutMs,
      () => stall("headers"),
    );
    if (!response.body) return response;

    const reader = response.body.getReader();
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const { done, value } = await withDeadline(
          reader.read(),
          timeoutMs,
          () => stall("body"),
        );
        if (done) controller.close();
        else controller.enqueue(value);
      },
      cancel(reason) {
        return reader.cancel(reason);
      },
    });
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
  return Object.assign(stallFetch, {
    preconnect: globalThis.fetch.preconnect,
  });
}
