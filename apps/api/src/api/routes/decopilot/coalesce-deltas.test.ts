import { describe, expect, test } from "bun:test";
import type { UIMessageChunk } from "ai";
import { coalesceDeltas } from "./coalesce-deltas";

const text = (id: string, delta: string): UIMessageChunk => ({
  type: "text-delta",
  id,
  delta,
});
const reasoning = (id: string, delta: string): UIMessageChunk => ({
  type: "reasoning-delta",
  id,
  delta,
});

async function* from(
  chunks: Array<UIMessageChunk | { waitMs: number } | Error>,
): AsyncGenerator<UIMessageChunk> {
  for (const c of chunks) {
    if (c instanceof Error) throw c;
    if ("waitMs" in c) {
      await new Promise((r) => setTimeout(r, c.waitMs));
      continue;
    }
    yield c;
  }
}

async function collect(source: AsyncIterable<UIMessageChunk>) {
  const out: UIMessageChunk[] = [];
  for await (const c of source) out.push(c);
  return out;
}

describe("coalesceDeltas", () => {
  test("joins a burst of deltas of the same part into one chunk", async () => {
    const out = await collect(
      coalesceDeltas(
        from([
          { type: "text-start", id: "t" },
          text("t", "Hel"),
          text("t", "lo "),
          text("t", "world"),
          { type: "text-end", id: "t" },
        ]),
        50,
      ),
    );
    expect(out).toEqual([
      { type: "text-start", id: "t" },
      text("t", "Hello world"),
      { type: "text-end", id: "t" },
    ]);
  });

  test("never joins across parts or delta kinds, and keeps order", async () => {
    const out = await collect(
      coalesceDeltas(
        from([
          reasoning("r", "think"),
          reasoning("r", "ing"),
          text("a", "x"),
          text("b", "y"),
          text("b", "z"),
        ]),
        50,
      ),
    );
    expect(out).toEqual([
      reasoning("r", "thinking"),
      text("a", "x"),
      text("b", "yz"),
    ]);
  });

  test("flushes on the timer when the model pauses mid-part", async () => {
    const seen: Array<{ chunk: UIMessageChunk; at: number }> = [];
    const start = Date.now();
    for await (const chunk of coalesceDeltas(
      from([text("t", "a"), text("t", "b"), { waitMs: 200 }, text("t", "c")]),
      20,
    )) {
      seen.push({ chunk, at: Date.now() - start });
    }
    expect(seen.map((s) => s.chunk)).toEqual([text("t", "ab"), text("t", "c")]);
    // "ab" went out on the 20ms window, not when "c" arrived ~200ms later.
    expect(seen[0]!.at).toBeLessThan(150);
  });

  test("passes a delta with provider metadata through on its own", async () => {
    const signed: UIMessageChunk = {
      type: "reasoning-delta",
      id: "r",
      delta: "",
      providerMetadata: { anthropic: { signature: "sig" } },
    };
    const out = await collect(
      coalesceDeltas(
        from([reasoning("r", "a"), signed, reasoning("r", "b")]),
        50,
      ),
    );
    expect(out).toEqual([reasoning("r", "a"), signed, reasoning("r", "b")]);
  });

  test("a window of 0 is a passthrough", async () => {
    const chunks = [text("t", "a"), text("t", "b")];
    expect(await collect(coalesceDeltas(from(chunks), 0))).toEqual(chunks);
  });

  test("text already received is emitted before a source error", async () => {
    const out: UIMessageChunk[] = [];
    let caught: unknown;
    try {
      for await (const c of coalesceDeltas(
        from([text("t", "a"), text("t", "b"), new Error("boom")]),
        1_000,
      )) {
        out.push(c);
      }
    } catch (err) {
      caught = err;
    }
    expect(out).toEqual([text("t", "ab")]);
    expect((caught as Error).message).toBe("boom");
  });

  test("stopping early releases the source", async () => {
    let released = false;
    async function* source(): AsyncGenerator<UIMessageChunk> {
      try {
        yield { type: "text-start", id: "t" };
        yield text("t", "a");
        yield { type: "text-end", id: "t" };
      } finally {
        released = true;
      }
    }
    for await (const c of coalesceDeltas(source(), 50)) {
      if (c.type === "text-start") break;
    }
    expect(released).toBe(true);
  });
});
