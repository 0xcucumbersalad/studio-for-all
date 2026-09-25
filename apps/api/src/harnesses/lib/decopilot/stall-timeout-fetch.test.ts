import { describe, expect, it } from "bun:test";
import {
  createStallTimeoutFetch,
  ProviderStallError,
  stallTimeoutMsFromEnv,
} from "./stall-timeout-fetch";

const TIMEOUT_MS = 50;

/** A fetch that never answers until its signal aborts — LiteLLM holding the request. */
const neverAnswers = ((_input: RequestInfo | URL, init?: RequestInit) =>
  new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
  })) as typeof fetch;

/** Sends the given chunks, then goes silent without closing the body. */
function stallsAfter(chunks: string[]): {
  fetch: typeof fetch;
  signals: AbortSignal[];
} {
  const signals: AbortSignal[] = [];
  const fetchImpl = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.signal) signals.push(init.signal);
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      },
    });
    return new Response(body, { status: 200 });
  }) as typeof fetch;
  return { fetch: fetchImpl, signals };
}

describe("createStallTimeoutFetch", () => {
  it("fails with a retryable network code when no response arrives", async () => {
    const stallFetch = createStallTimeoutFetch(TIMEOUT_MS, neverAnswers);
    const error = await stallFetch("http://litellm.test").catch((e) => e);
    expect(error).toBeInstanceOf(ProviderStallError);
    // The AI SDK retries fetch errors carrying this code.
    expect(error.code).toBe("UND_ERR_HEADERS_TIMEOUT");
    expect(error.message).toContain("LLM provider stalled");
  });

  it("fails a stream that goes silent mid-body and aborts the request", async () => {
    const { fetch: stalling, signals } = stallsAfter(["data: a\n\n"]);
    const stallFetch = createStallTimeoutFetch(TIMEOUT_MS, stalling);
    const response = await stallFetch("http://litellm.test");
    const reader = response.body!.getReader();
    const first = await reader.read();
    expect(new TextDecoder().decode(first.value)).toBe("data: a\n\n");
    const error = await reader.read().catch((e) => e);
    expect(error).toBeInstanceOf(ProviderStallError);
    expect(error.code).toBeUndefined();
    expect(signals[0]?.aborted).toBe(true);
  });

  it("passes a healthy response through unchanged", async () => {
    const healthy = (async () =>
      new Response("data: [DONE]\n\n", {
        status: 201,
        headers: { "Content-Type": "text/event-stream" },
      })) as unknown as typeof fetch;
    const stallFetch = createStallTimeoutFetch(TIMEOUT_MS, healthy);
    const response = await stallFetch("http://litellm.test");
    expect(response.status).toBe(201);
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
    expect(await response.text()).toBe("data: [DONE]\n\n");
  });

  it("only times provider silence, not a slow consumer", async () => {
    const { fetch: stalling } = stallsAfter(["a", "b"]);
    const stallFetch = createStallTimeoutFetch(TIMEOUT_MS, stalling);
    const response = await stallFetch("http://litellm.test");
    const reader = response.body!.getReader();
    await reader.read();
    // The consumer (e.g. a long tool call) holds the stream past the timeout.
    await Bun.sleep(TIMEOUT_MS * 3);
    const second = await reader.read();
    expect(new TextDecoder().decode(second.value)).toBe("b");
    await reader.cancel();
  });

  it("keeps the caller's abort as the abort reason", async () => {
    const stallFetch = createStallTimeoutFetch(10_000, neverAnswers);
    const caller = new AbortController();
    const pending = stallFetch("http://litellm.test", {
      signal: caller.signal,
    }).catch((e) => e);
    caller.abort(new DOMException("cancelled", "AbortError"));
    const error = await pending;
    expect(error).not.toBeInstanceOf(ProviderStallError);
    expect(error.name).toBe("AbortError");
  });
});

describe("stallTimeoutMsFromEnv", () => {
  it("defaults when unset, empty, or invalid", () => {
    for (const value of [
      undefined,
      "",
      "  ",
      "abc",
      "-5",
      "1.5e",
      "Infinity",
    ]) {
      expect(stallTimeoutMsFromEnv(value)).toBe(120_000);
    }
  });

  it("reads a positive value and treats 0 as off", () => {
    expect(stallTimeoutMsFromEnv("30000")).toBe(30_000);
    expect(stallTimeoutMsFromEnv(" 0 ")).toBe(0);
  });
});
