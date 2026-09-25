import { describe, expect, it } from "bun:test";
import { runNativeAgentLoopCore } from "./native-agent-loop-core";

describe("runNativeAgentLoopCore", () => {
  it("passes shared stream options through and captures provider errors", async () => {
    let capturedConfig:
      | {
          messages: unknown[];
          tools: Record<string, unknown>;
          temperature: number;
          maxOutputTokens: number;
          onError: (event: { error: unknown }) => void | Promise<void>;
        }
      | undefined;

    const fakeResult = {
      finishReason: Promise.resolve("error"),
      toUIMessageStream: () => ({
        async *[Symbol.asyncIterator]() {},
      }),
    };

    const handle = runNativeAgentLoopCore({
      model: { specificationVersion: "v2" } as never,
      systemMessages: [{ role: "system", content: "system" }],
      messages: [{ role: "user", content: "hi" }] as never,
      tools: { user_ask: {} as never } as never,
      temperature: 0.2,
      maxOutputTokens: 1234,
      stopWhen: () => false,
      abortSignal: new AbortController().signal,
      streamText: (config) => {
        capturedConfig = config as typeof capturedConfig;
        Promise.resolve().then(() =>
          capturedConfig?.onError({ error: new Error("provider exploded") }),
        );
        return fakeResult as never;
      },
    });

    expect(capturedConfig?.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(Object.keys(capturedConfig?.tools ?? {})).toEqual(["user_ask"]);
    expect(capturedConfig?.temperature).toBe(0.2);
    expect(capturedConfig?.maxOutputTokens).toBe(1234);
    await expect(handle.error).resolves.toContain("provider exploded");
  });

  // The AI SDK wraps a mid-stream body failure; the reason is only on `cause`.
  it("keeps a wrapped provider error's cause in the message", async () => {
    const handle = runNativeAgentLoopCore({
      model: { specificationVersion: "v2" } as never,
      systemMessages: [],
      messages: [],
      tools: {},
      maxOutputTokens: 1,
      stopWhen: () => false,
      abortSignal: new AbortController().signal,
      streamText: (config) => {
        const onError = (
          config as { onError: (event: { error: unknown }) => unknown }
        ).onError;
        Promise.resolve().then(() =>
          onError({
            error: new Error("Failed to process successful response", {
              cause: new Error(
                "LLM provider stalled: stream sent no data for 120s",
              ),
            }),
          }),
        );
        return { finishReason: Promise.resolve("error") } as never;
      },
    });

    await expect(handle.error).resolves.toBe(
      "Failed to process successful response: LLM provider stalled: stream sent no data for 120s",
    );
  });
});
