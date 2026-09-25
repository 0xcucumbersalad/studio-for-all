import { describe, expect, mock, test } from "bun:test";
import type { LanguageModelV4 } from "@ai-sdk/provider";
import { Hono } from "hono";
import type { Env } from "@/api/hono-env";
import type { StudioContext } from "@/core/studio-context";
import {
  buildImprovePrompt,
  createInstructionsImproveRoutes,
  ImproveInstructionsSchema,
} from "./instructions-improve";

/** A model that streams `chunks` as text and records what it was asked. */
function streamingModel(chunks: string[]) {
  const calls: unknown[] = [];
  const model = {
    specificationVersion: "v4",
    provider: "test",
    modelId: "test-model",
    supportedUrls: {},
    doGenerate: async () => {
      throw new Error("unused");
    },
    doStream: async (options: unknown) => {
      calls.push(options);
      return {
        stream: new ReadableStream({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings: [] });
            controller.enqueue({ type: "text-start", id: "t" });
            for (const delta of chunks) {
              controller.enqueue({ type: "text-delta", id: "t", delta });
            }
            controller.enqueue({ type: "text-end", id: "t" });
            controller.enqueue({
              type: "finish",
              finishReason: { unified: "stop", raw: "stop" },
              usage: {
                inputTokens: { total: 1 },
                outputTokens: { total: 1 },
              },
            });
            controller.close();
          },
        }),
      };
    },
  } as unknown as LanguageModelV4;
  return { model, calls };
}

function makeApp(opts: {
  user?: boolean;
  org?: boolean;
  model?: LanguageModelV4;
  noKeys?: boolean;
}) {
  const activate = mock(() =>
    Promise.resolve({
      aiSdk: { languageModel: () => opts.model ?? streamingModel([]).model },
    }),
  );
  const ctx = {
    organization: opts.org === false ? undefined : { id: "org_1" },
    auth: { user: opts.user === false ? undefined : { id: "user_1" } },
    storage: {
      organizationSettings: {
        get: mock(() =>
          Promise.resolve({
            simple_mode: { tiers: { smart: { keyId: "k1", modelId: "m" } } },
          }),
        ),
      },
      userModelPreferences: { get: mock(() => Promise.resolve(null)) },
      aiProviderKeys: {
        list: mock(() =>
          Promise.resolve(
            opts.noKeys
              ? []
              : [
                  {
                    id: "k1",
                    providerId: "openai-compatible",
                    label: "",
                    presetId: null,
                    createdBy: "u",
                    createdAt: "2026-01-01",
                  },
                ],
          ),
        ),
      },
    },
    aiProviders: { listModels: mock(() => Promise.resolve([])), activate },
  } as unknown as StudioContext;

  const app = new Hono<Env>();
  app.use("*", async (c, next) => {
    c.set("studioContext", ctx);
    await next();
  });
  app.route("/", createInstructionsImproveRoutes());
  return { app, activate };
}

const post = (app: Hono<Env>, body: unknown) =>
  app.request("/instructions/improve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

describe("ImproveInstructionsSchema", () => {
  test("accepts both kinds and trims", () => {
    expect(
      ImproveInstructionsSchema.parse({ kind: "agent", instructions: "  x " })
        .instructions,
    ).toBe("x");
  });

  test("rejects empty, whitespace-only, oversized and unknown kinds", () => {
    for (const body of [
      { kind: "agent", instructions: "" },
      { kind: "agent", instructions: "   " },
      { kind: "agent", instructions: "x".repeat(20_001) },
      { kind: "workflow", instructions: "x" },
      { kind: "agent" },
    ]) {
      expect(ImproveInstructionsSchema.safeParse(body).success).toBe(false);
    }
  });
});

describe("buildImprovePrompt", () => {
  test("names the subject and fences the current instructions", () => {
    const prompt = buildImprovePrompt({
      kind: "automation",
      name: "Daily digest",
      instructions: "summarize issues",
    });
    expect(prompt).toContain('automation "Daily digest"');
    expect(prompt).toContain(
      "<current_instructions>\nsummarize issues\n</current_instructions>",
    );
  });
});

describe("POST /instructions/improve", () => {
  test("streams the model's rewrite as plain text, in one model call", async () => {
    const { model, calls } = streamingModel([
      "<role>",
      "\nA helper",
      "</role>",
    ]);
    const { app, activate } = makeApp({ model });
    const res = await post(app, {
      kind: "automation",
      name: "Digest",
      instructions: "do the thing",
    });
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("<role>\nA helper</role>");
    expect(calls).toHaveLength(1);
    expect(activate).toHaveBeenCalledWith("k1", "org_1");
    // The automation system prompt, not the agent one.
    expect(JSON.stringify(calls[0])).toContain("nobody watching");
    expect(JSON.stringify(calls[0])).toContain("do the thing");
  });

  test("400 on an invalid body, before any model is resolved", async () => {
    const { app, activate } = makeApp({});
    const res = await post(app, { kind: "agent", instructions: "" });
    expect(res.status).toBe(400);
    expect(activate).not.toHaveBeenCalled();
  });

  test("401 without a signed-in principal", async () => {
    const { app } = makeApp({ user: false });
    expect((await post(app, { kind: "agent", instructions: "x" })).status).toBe(
      401,
    );
  });

  test("400 without an organization", async () => {
    const { app } = makeApp({ org: false });
    expect((await post(app, { kind: "agent", instructions: "x" })).status).toBe(
      400,
    );
  });

  test("400 with a readable message when no model is configured", async () => {
    const { app } = makeApp({ noKeys: true });
    const res = await post(app, { kind: "agent", instructions: "x" });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toMatch(
      /No model available/,
    );
  });
});
