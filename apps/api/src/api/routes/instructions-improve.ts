/**
 * POST /api/:org/instructions/improve — rewrite an agent's or automation's
 * instructions in ONE streaming model call.
 *
 * The Improve buttons used to send a chat message that delegated to a manager
 * agent, which then read two docs, fetched the record it had just been handed,
 * wrote the rewrite as tool input, saved it and re-read it — six or seven
 * sequential model calls, each resending the conversation and every tool
 * schema, before anything reached the editor. This is the same rewrite as a
 * single call whose text streams straight into the editor; the page saves it
 * through its own autosave (and the user can undo), so nothing here writes.
 *
 * Mounted under `/api/:org`, so `resolveOrgFromPath` has already checked
 * membership. It spends AI credit, so the budget gate applies, like chat.
 */

import { streamText } from "ai";
import { Hono } from "hono";
import { z } from "zod";
import type { Env } from "@/api/hono-env";
import {
  AiBudgetExhaustedError,
  assertAiBudget,
} from "@/core/plan-feature-gate";
import { resolveTier, TierUnavailableError } from "@/core/resolve-tier";

/** Same ceiling the prompt editor uses — rejects a runaway body, nothing more. */
const MAX_INSTRUCTIONS_LENGTH = 20_000;

export const ImproveInstructionsSchema = z.object({
  kind: z.enum(["agent", "automation"]),
  instructions: z.string().trim().min(1).max(MAX_INSTRUCTIONS_LENGTH),
  /** The agent's or automation's display name — context for the rewrite. */
  name: z.string().trim().max(200).optional(),
});

export type ImproveInstructionsInput = z.infer<
  typeof ImproveInstructionsSchema
>;

const SHARED_RULES = `<rules>
- Rewrite the instructions using explicit XML-style sections, in this order: <role>, <capabilities>, <constraints>, <workflows>.
- Keep the author's intent, scope, names, tools, URLs, identifiers and examples exactly. Never invent capabilities, tools, data sources or facts that are not in the original.
- If a workflow exists, sharpen it into concrete, ordered, operational steps. If none exists, add one that follows from the original.
- Tighten <constraints> when the original is too open-ended.
- Write in the same language as the original.
- Output ONLY the rewritten instructions: no preamble, no commentary, no code fences.
</rules>`;

const SYSTEM: Record<ImproveInstructionsInput["kind"], string> = {
  agent: `You improve the instructions (system prompt) of an AI agent that people chat with.

${SHARED_RULES}`,
  automation: `You improve the instructions of an automation: a background agent run started by a trigger (a cron schedule or an event), with nobody watching.

${SHARED_RULES}
<automation_rules>
- The run cannot ask anyone questions. Make every step executable without clarification, and say what to do when data is missing.
- State what the run must produce or change by the end, so its result is checkable.
</automation_rules>`,
};

/** The prompt sent with the system message. Pure — unit-tested. */
export function buildImprovePrompt(input: ImproveInstructionsInput): string {
  const subject = input.name ? `${input.kind} "${input.name}"` : input.kind;
  return (
    `Improve the instructions of this ${subject}.\n` +
    `<current_instructions>\n${input.instructions}\n</current_instructions>`
  );
}

export function createInstructionsImproveRoutes(): Hono<Env> {
  const app = new Hono<Env>();

  app.post("/instructions/improve", async (c) => {
    const ctx = c.get("studioContext");
    const orgId = ctx.organization?.id;
    if (!orgId) return c.json({ error: "Organization required" }, 400);
    if (!ctx.auth.user?.id && !ctx.auth.apiKey?.id) {
      return c.json({ error: "Unauthorized" }, 401);
    }

    const parsed = ImproveInstructionsSchema.safeParse(
      await c.req.json().catch(() => null),
    );
    if (!parsed.success) {
      return c.json(
        { error: parsed.error.issues[0]?.message ?? "Invalid request" },
        400,
      );
    }

    try {
      await assertAiBudget(ctx, orgId, "Improving instructions");
    } catch (err) {
      if (err instanceof AiBudgetExhaustedError) {
        return c.json({ error: err.message }, 402);
      }
      throw err;
    }

    let model;
    try {
      const tier = await resolveTier(ctx, "smart");
      const provider = await ctx.aiProviders.activate(tier.credentialId, orgId);
      model = provider.aiSdk.languageModel(tier.modelId);
    } catch (err) {
      if (err instanceof TierUnavailableError) {
        return c.json({ error: err.message }, 400);
      }
      throw err;
    }

    const result = streamText({
      model,
      instructions: SYSTEM[parsed.data.kind],
      prompt: buildImprovePrompt(parsed.data),
      temperature: 0.3,
      // Closing the page stops the model, not just the response.
      abortSignal: c.req.raw.signal,
      // A text stream has no error frame; the page treats an empty result as
      // a failure, and the cause is logged here.
      onError: ({ error }) => {
        console.error("[instructions-improve] stream failed", {
          message: error instanceof Error ? error.message : String(error),
        });
      },
    });
    return result.toTextStreamResponse();
  });

  return app;
}
