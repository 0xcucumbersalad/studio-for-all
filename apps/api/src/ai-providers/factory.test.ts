import { afterEach, describe, expect, test } from "bun:test";
import type { AIProviderKeyStorage } from "../storage/ai-provider-keys";
import { AIProviderFactory, resetModelListStateForTest } from "./factory";
import { InMemoryModelListCache } from "./model-list-cache";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  resetModelListStateForTest();
});

const GOOGLE_MODELS_BODY = {
  models: [
    {
      name: "models/gemini-2.5-flash",
      displayName: "Gemini 2.5 Flash",
      version: "1",
      inputTokenLimit: 100,
      outputTokenLimit: 10,
      supportedGenerationMethods: ["generateContent"],
      thinking: false,
      temperature: 1,
      maxTemperature: 1,
      description: "d",
      topP: 1,
      topK: 1,
    },
  ],
};

// OpenRouter omits `supported_parameters` for some models.
const OPENROUTER_MODELS_BODY = {
  data: [
    {
      id: "google/gemini-2.5-flash",
      canonical_slug: "google/gemini-2.5-flash",
      name: "Gemini 2.5 Flash",
      created: 0,
      // OpenRouter serializes pricing as decimal strings, not numbers.
      pricing: {
        prompt: "0.0000005808",
        completion: "0.0000017424",
        request: "0",
        image: "0",
      },
      context_length: 100,
      architecture: {
        modality: "text",
        input_modalities: ["text", "image"],
        output_modalities: ["text"],
        tokenizer: "x",
      },
      top_provider: {
        is_moderated: false,
        context_length: 100,
        max_completion_tokens: 10,
      },
      description: "d",
    },
  ],
};

function fakeStorage(): AIProviderKeyStorage {
  return {
    resolve: async () => ({
      keyInfo: { id: "key-1", providerId: "google" } as never,
      apiKey: "secret",
    }),
  } as unknown as AIProviderKeyStorage;
}

describe("AIProviderFactory.listModels", () => {
  test("enriches from OpenRouter even when a model omits supported_parameters", async () => {
    globalThis.fetch = (async (url: unknown): Promise<Response> => {
      const u = String(url);
      const parsed = new URL(u);
      if (parsed.hostname === "generativelanguage.googleapis.com") {
        return new Response(JSON.stringify(GOOGLE_MODELS_BODY), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (
        parsed.hostname === "openrouter.ai" &&
        parsed.pathname === "/api/v1/models"
      ) {
        return new Response(JSON.stringify(OPENROUTER_MODELS_BODY), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch: ${u}`);
    }) as unknown as typeof fetch;

    const factory = new AIProviderFactory(fakeStorage());
    const models = await factory.listModels("key-1", "org-1");

    expect(models).toHaveLength(1);
    // Regression: this model previously crashed index-building, blanking enrichment.
    expect(models[0]?.capabilities).toContain("vision");
    // Regression: OpenRouter's decimal-string pricing must be parsed to numbers.
    expect(models[0]?.costs?.input).toBe(0.0000005808);
    expect(models[0]?.costs?.output).toBe(0.0000017424);
  });

  test("retries a transient 503 fetching the OpenRouter enrichment index instead of blanking enrichment", async () => {
    let openRouterCalls = 0;
    globalThis.fetch = (async (url: unknown): Promise<Response> => {
      const u = String(url);
      const parsed = new URL(u);
      if (parsed.hostname === "generativelanguage.googleapis.com") {
        return new Response(JSON.stringify(GOOGLE_MODELS_BODY), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (
        parsed.hostname === "openrouter.ai" &&
        parsed.pathname === "/api/v1/models"
      ) {
        openRouterCalls++;
        if (openRouterCalls === 1) {
          return new Response("upstream hiccup", { status: 503 });
        }
        return new Response(JSON.stringify(OPENROUTER_MODELS_BODY), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch: ${u}`);
    }) as unknown as typeof fetch;

    const factory = new AIProviderFactory(fakeStorage());
    const models = await factory.listModels("key-1", "org-1");

    expect(openRouterCalls).toBeGreaterThan(1);
    expect(models[0]?.capabilities).toContain("vision");
  });

  test("defaults OpenRouter enrichment costs to 0 when pricing is malformed, not NaN", async () => {
    globalThis.fetch = (async (url: unknown): Promise<Response> => {
      const u = String(url);
      const parsed = new URL(u);
      if (parsed.hostname === "generativelanguage.googleapis.com") {
        return new Response(JSON.stringify(GOOGLE_MODELS_BODY), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (
        parsed.hostname === "openrouter.ai" &&
        parsed.pathname === "/api/v1/models"
      ) {
        const body = {
          data: [
            {
              ...OPENROUTER_MODELS_BODY.data[0],
              pricing: { prompt: "", completion: "", request: "0", image: "0" },
            },
          ],
        };
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`unexpected fetch: ${u}`);
    }) as unknown as typeof fetch;

    const factory = new AIProviderFactory(fakeStorage());
    const models = await factory.listModels("key-1", "org-1");

    expect(models[0]?.costs?.input).toBe(0);
    expect(models[0]?.costs?.output).toBe(0);
  });
});

describe("AIProviderFactory.listModels caching", () => {
  /** Two openai-compatible keys pointing at two different servers. */
  function compatStorage(): AIProviderKeyStorage {
    return {
      resolve: async (keyId: string) => ({
        keyInfo: { id: keyId, providerId: "openai-compatible" } as never,
        apiKey: JSON.stringify({
          baseUrl: `http://${keyId}.test`,
          apiKey: "k",
        }),
      }),
    } as unknown as AIProviderKeyStorage;
  }

  /** Records upstream calls; `/models` answers with one model named after the host. */
  function fakeUpstream(opts: { failModels?: boolean } = {}) {
    const calls: string[] = [];
    globalThis.fetch = (async (url: unknown): Promise<Response> => {
      const u = new URL(String(url));
      calls.push(u.host + u.pathname);
      if (u.hostname === "openrouter.ai") {
        return new Response("down", { status: 503 });
      }
      if (opts.failModels) return new Response("nope", { status: 401 });
      return new Response(
        JSON.stringify({ data: [{ id: `model-of-${u.hostname}` }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;
    return calls;
  }

  test("caches per key, so two openai-compatible keys keep their own lists", async () => {
    fakeUpstream();
    const factory = new AIProviderFactory(
      compatStorage(),
      new InMemoryModelListCache(),
    );
    const a = await factory.listModels("a", "org");
    const b = await factory.listModels("b", "org");
    expect(a.map((m) => m.modelId)).toEqual(["model-of-a.test"]);
    expect(b.map((m) => m.modelId)).toEqual(["model-of-b.test"]);
  });

  test("concurrent listings of one key share a single upstream call", async () => {
    const calls = fakeUpstream();
    const factory = new AIProviderFactory(compatStorage());
    await Promise.all([
      factory.listModels("a", "org"),
      factory.listModels("a", "org"),
      factory.listModels("a", "org"),
    ]);
    expect(calls.filter((c) => c === "a.test/v1/models")).toHaveLength(1);
  });

  test("a failed listing is remembered instead of re-fetched every turn", async () => {
    const calls = fakeUpstream({ failModels: true });
    const factory = new AIProviderFactory(compatStorage());
    await expect(factory.listModels("a", "org")).rejects.toThrow();
    const after = calls.length;
    await expect(factory.listModels("a", "org")).rejects.toThrow();
    expect(calls.length).toBe(after);
  });

  test("a failed OpenRouter enrichment is not retried on the next listing", async () => {
    const calls = fakeUpstream();
    const orCalls = () =>
      calls.filter((c) => c.startsWith("openrouter.ai")).length;
    const factory = new AIProviderFactory(compatStorage());
    await factory.listModels("a", "org");
    const afterFirst = orCalls();
    expect(afterFirst).toBeGreaterThan(0);
    // Another key's listing still works, without asking openrouter.ai again.
    const b = await factory.listModels("b", "org");
    expect(b.map((m) => m.modelId)).toEqual(["model-of-b.test"]);
    expect(orCalls()).toBe(afterFirst);
  });
});
