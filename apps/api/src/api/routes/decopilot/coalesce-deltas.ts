/**
 * Merge consecutive text/reasoning deltas before they are published.
 *
 * `ingestRun` publishes every chunk to JetStream and waits for the ack before
 * pulling the next one — which is what makes the run's log durable and
 * resumable. An OpenAI-compatible gateway (LiteLLM, a local server) sends one
 * SSE event per token, so a turn is thousands of chunks and every one of them
 * paid a replicated, file-backed ack in series: the model got ahead of the
 * publisher and the text reached the browser late and in bursts.
 *
 * Deltas of the same part (same `type` and `id`) are joined for at most
 * `windowMs` after the first one arrives, then published as one chunk. The
 * window is a real timer, not "until the next chunk": a model that pauses
 * mid-sentence still gets its last words out on time. Anything else — a part
 * boundary, a tool call, a delta carrying provider metadata (a reasoning
 * signature must stay on its own chunk) — flushes first and passes through
 * untouched, so ordering and part structure are exactly what the model sent.
 */

import type { UIMessageChunk } from "ai";

type DeltaChunk = Extract<
  UIMessageChunk,
  { type: "text-delta" } | { type: "reasoning-delta" }
>;

function isMergeableDelta(chunk: UIMessageChunk): chunk is DeltaChunk {
  return (
    (chunk.type === "text-delta" || chunk.type === "reasoning-delta") &&
    chunk.providerMetadata === undefined
  );
}

const TIMEOUT = Symbol("timeout");

export async function* coalesceDeltas(
  source: AsyncIterable<UIMessageChunk>,
  windowMs: number,
): AsyncGenerator<UIMessageChunk> {
  if (windowMs <= 0) {
    yield* source;
    return;
  }
  const iterator = source[Symbol.asyncIterator]();
  let pending: Promise<IteratorResult<UIMessageChunk>> | null = null;
  // Held on an object: TypeScript's narrowing of a `let` across the awaits
  // and yields below collapses it to `never`.
  const held: { chunk: DeltaChunk | null } = { chunk: null };
  let bufferedAt = 0;
  let finished = false;

  try {
    for (;;) {
      pending ??= iterator.next();
      let result: IteratorResult<UIMessageChunk> | typeof TIMEOUT;
      if (held.chunk) {
        const remaining = windowMs - (Date.now() - bufferedAt);
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          result =
            remaining <= 0
              ? TIMEOUT
              : await Promise.race([
                  pending,
                  new Promise<typeof TIMEOUT>((resolve) => {
                    timer = setTimeout(() => resolve(TIMEOUT), remaining);
                  }),
                ]);
        } catch (error) {
          // The source failed: what it had already said still goes out first.
          pending = null;
          const text = held.chunk;
          held.chunk = null;
          yield text;
          throw error;
        } finally {
          if (timer) clearTimeout(timer);
        }
        if (result === TIMEOUT) {
          // The pending pull stays outstanding and is awaited next iteration.
          yield held.chunk;
          held.chunk = null;
          continue;
        }
      } else {
        result = await pending;
      }
      pending = null;

      if (result.done) {
        finished = true;
        break;
      }
      const chunk = result.value;
      if (isMergeableDelta(chunk)) {
        const head = held.chunk;
        if (head && head.type === chunk.type && head.id === chunk.id) {
          held.chunk = { ...head, delta: head.delta + chunk.delta };
          continue;
        }
        if (held.chunk) yield held.chunk;
        held.chunk = chunk;
        bufferedAt = Date.now();
        continue;
      }
      if (held.chunk) {
        yield held.chunk;
        held.chunk = null;
      }
      yield chunk;
    }
    if (held.chunk) yield held.chunk;
  } finally {
    // Stopped early (consumer broke out, or a downstream error): release the
    // source like a plain for-await would. With a pull still in flight,
    // awaiting `return()` would wait on that pull (a generator queues it), so
    // it is fired and left, and the orphaned pull's outcome is swallowed.
    if (!finished) {
      const released = Promise.resolve(iterator.return?.()).catch(() => {});
      if (pending) pending.catch(() => {});
      else await released;
    }
  }
}
