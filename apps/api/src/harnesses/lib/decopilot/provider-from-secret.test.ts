import { describe, expect, it } from "bun:test";
import type { DecopilotSecretModelSource } from "../sources";
import { createProviderFromSecret } from "./provider-from-secret";

const secret = (providerId: string): DecopilotSecretModelSource =>
  ({
    kind: "secret",
    providerId,
    apiKey: "sk-test-unused-no-network",
    modelId: "x/y",
  }) as DecopilotSecretModelSource;

describe("createProviderFromSecret", () => {
  for (const providerId of ["openrouter", "deco"] as const) {
    it(`${providerId}: exposes a native decision model independently of chat`, () => {
      const provider = createProviderFromSecret(secret(providerId));
      const model = provider.decisions?.model("typesafe/jev-1.13");
      expect(model).toMatchObject({
        modelId: "typesafe/jev-1.13",
        doEvaluate: expect.any(Function),
      });
    });
  }

  // Regression guard: the openrouter/deco branch once wrapped languageModel as
  // `(...args) => aiSdk.languageModel(...args)` AFTER `Object.assign` had already
  // replaced `aiSdk.languageModel` with that very wrapper — a self-referential
  // tail call. Under JSC (Bun) proper-tail-call elimination this becomes a 100%
  // CPU infinite loop (not a stack overflow), so the desktop sandbox daemon
  // wedged on every decopilot run. `languageModel(...)` must construct a model
  // synchronously and return. (A regression would hang this test — a loud,
  // CI-visible failure, since a synchronous infinite loop ignores test timeouts.)
  for (const providerId of [
    "openrouter",
    "deco",
    "llmapi",
    "openai-compatible",
  ] as const) {
    it(`${providerId}: languageModel returns a model without self-recursion`, () => {
      const provider = createProviderFromSecret(secret(providerId));
      const model = provider.aiSdk.languageModel("some-model");
      expect(model).toBeDefined();
      expect(typeof (model as { modelId?: unknown }).modelId).toBe("string");
    });
  }
});

describe("openai-compatible chat streaming", () => {
  /** LiteLLM-style SSE: reasoning in `reasoning_content`, then the answer. */
  function sse(events: unknown[]): Response {
    const body =
      events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") +
      "data: [DONE]\n\n";
    return new Response(body, {
      headers: { "Content-Type": "text/event-stream" },
    });
  }
  const delta = (d: Record<string, unknown>, finish: string | null = null) => ({
    id: "c1",
    object: "chat.completion.chunk",
    created: 1,
    model: "m",
    choices: [{ index: 0, delta: d, finish_reason: finish }],
  });

  it("streams reasoning_content as reasoning, not silence", async () => {
    const realFetch = globalThis.fetch;
    const requests: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      requests.push(String(input));
      return sse([
        delta({ role: "assistant", reasoning_content: "Let me " }),
        delta({ reasoning_content: "think." }),
        delta({ content: "Hi" }),
        delta({}, "stop"),
      ]);
    }) as typeof fetch;
    try {
      const { streamText } = await import("ai");
      const provider = createProviderFromSecret({
        ...secret("openai-compatible"),
        baseUrl: "http://litellm.test",
      } as DecopilotSecretModelSource);
      const result = streamText({
        model: provider.aiSdk.languageModel("m"),
        prompt: "hi",
      });
      const kinds: string[] = [];
      for await (const part of result.fullStream) kinds.push(part.type);
      expect(await result.reasoningText).toBe("Let me think.");
      expect(await result.text).toBe("Hi");
      // Reasoning arrived as its own parts, before the text.
      expect(kinds.indexOf("reasoning-delta")).toBeLessThan(
        kinds.indexOf("text-delta"),
      );
      expect(requests[0]).toBe("http://litellm.test/v1/chat/completions");
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
