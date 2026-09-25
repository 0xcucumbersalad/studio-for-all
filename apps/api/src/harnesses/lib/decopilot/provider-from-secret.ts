/**
 * Build the provider surface Decopilot consumes from resolved secret model
 * sources. It stays independent of cluster provider, vault, storage, and
 * `ai-providers/*` imports so the hosted core remains easy to test.
 */

import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogle } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createOpenAI } from "@ai-sdk/openai";
import type { ProviderV4 } from "@ai-sdk/provider";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import {
  isInteractionsOnlyModel,
  pollInteraction,
  submitInteraction,
} from "./gemini-interactions";
import type { DecopilotSecretModelSource } from "../types";
import type { StudioProvider } from "./studio-provider";

export interface ResolvedSecretProvider extends StudioProvider {
  info: { id: never; name: string; description: string };
  aiSdk: ProviderV4;
  listModels(): Promise<never[]>;
}

function withProviderSurface(
  source: DecopilotSecretModelSource,
  aiSdk: ProviderV4,
  extras: Pick<ResolvedSecretProvider, "asyncResearch" | "decisions"> = {},
): ResolvedSecretProvider {
  return {
    info: {
      id: source.providerId as never,
      name: source.providerId,
      description: source.providerId,
    },
    aiSdk,
    ...extras,
    listModels: async () => {
      throw new Error(
        `Decopilot provider '${source.providerId}' was created from a resolved runtime secret and cannot list models.`,
      );
    },
  };
}

/**
 * OPENAI_COMPATIBLE_STREAMS_REASONING (default on). Read from the environment
 * here rather than through `@/settings`: this module is portable harness code
 * and must stay free of app-local imports. `false` returns openai-compatible
 * chat to the `@ai-sdk/openai` client.
 */
function openaiCompatibleStreamsReasoning(): boolean {
  const value = process.env.OPENAI_COMPATIBLE_STREAMS_REASONING?.trim();
  return !value || value === "true" || value === "1";
}

export function createProviderFromSecret(
  source: DecopilotSecretModelSource,
): ResolvedSecretProvider {
  const { providerId, apiKey, baseUrl, extraHeaders } = source;

  switch (providerId) {
    case "anthropic":
      return withProviderSurface(
        source,
        createAnthropic({
          apiKey,
          ...(baseUrl ? { baseURL: baseUrl } : {}),
          ...(extraHeaders ? { headers: extraHeaders } : {}),
        }),
      );

    case "google":
      return withProviderSurface(
        source,
        createGoogle({
          apiKey,
          ...(baseUrl ? { baseURL: baseUrl } : {}),
          ...(extraHeaders ? { headers: extraHeaders } : {}),
        }),
        {
          asyncResearch: {
            canHandle: isInteractionsOnlyModel,
            start: async ({ modelId, query, abortSignal }) => {
              const { interactionId } = await submitInteraction({
                apiKey,
                agent: modelId,
                query,
                abortSignal,
              });
              return { jobId: interactionId };
            },
            resume: ({ jobId, abortSignal, onProgress, pollIntervalMs }) =>
              pollInteraction({
                apiKey,
                interactionId: jobId,
                abortSignal,
                onProgress,
                pollIntervalMs,
              }),
          },
        },
      );

    case "openrouter":
    case "deco": {
      const aiSdk = createOpenRouter({
        apiKey,
        ...(baseUrl ? { baseURL: baseUrl } : {}),
        ...(extraHeaders ? { headers: extraHeaders } : {}),
      });
      // Capture the ORIGINAL factory before the Object.assign below overwrites
      // `aiSdk.languageModel`. A wrapper of the form `(...a) => aiSdk.languageModel(...a)`
      // would, post-assign, call ITSELF — a self-referential tail call that JSC
      // (Bun) tail-call-eliminates into a 100% CPU infinite loop (not a stack
      // overflow), wedging the desktop sandbox daemon on every decopilot run.
      const baseLanguageModel = aiSdk.languageModel.bind(aiSdk);
      return withProviderSurface(
        source,
        Object.assign(aiSdk, {
          languageModel: (...args: Parameters<typeof aiSdk.languageModel>) =>
            baseLanguageModel(...args),
        }) as ProviderV4,
        { decisions: { model: (modelId) => aiSdk.evaluationModel(modelId) } },
      );
    }

    case "llmapi":
    case "openai-compatible": {
      // llmapi is a fixed-endpoint OpenAI-compatible gateway; openai-compatible
      // is user-configured. Both route languageModel() through chat completions.
      let normalizedBaseUrl = (baseUrl ?? "").replace(/\/+$/, "");
      if (providerId === "llmapi" && !normalizedBaseUrl) {
        normalizedBaseUrl = "https://api.llmapi.ai/v1";
      }
      if (normalizedBaseUrl && !normalizedBaseUrl.endsWith("/v1")) {
        normalizedBaseUrl += "/v1";
      }
      const openai = createOpenAI({
        apiKey: apiKey || "not-needed",
        name: providerId,
        ...(normalizedBaseUrl ? { baseURL: normalizedBaseUrl } : {}),
        ...(extraHeaders ? { headers: extraHeaders } : {}),
      });
      // Chat goes through `@ai-sdk/openai-compatible`: `@ai-sdk/openai`'s chat
      // model never reads `reasoning_content`, so a thinking model behind
      // LiteLLM / vLLM / Ollama streamed nothing while it thought and then
      // dumped its answer. The OpenAI provider stays for the other surfaces.
      const chatModel = openaiCompatibleStreamsReasoning()
        ? createOpenAICompatible({
            name: providerId,
            baseURL: normalizedBaseUrl || "https://api.openai.com/v1",
            apiKey: apiKey || "not-needed",
            ...(extraHeaders ? { headers: extraHeaders } : {}),
            // Token usage on the final chunk — what the OpenAI client sent too.
            includeUsage: true,
          }).chatModel
        : openai.chat;
      return withProviderSurface(
        source,
        Object.assign(openai, {
          languageModel: (modelId: string) => chatModel(modelId),
        }) as ProviderV4,
      );
    }

    default:
      throw new Error(
        `decopilot: unsupported modelSource.providerId '${providerId}'. ` +
          "Supported: anthropic, google, openrouter, deco, openai-compatible, llmapi.",
      );
  }
}
