/**
 * Client for `POST /api/:org/instructions/improve`: one model call whose
 * rewrite streams back as plain text, so the Improve buttons can fill the
 * editor as it arrives instead of routing through a delegated chat run.
 */

export interface ImproveInstructionsRequest {
  orgSlug: string;
  kind: "agent" | "automation";
  instructions: string;
  name?: string;
  /** Called with the full text so far on every chunk. */
  onText: (text: string) => void;
  signal?: AbortSignal;
}

/** Resolves with the final rewrite; rejects with the server's message. */
export async function streamImprovedInstructions({
  orgSlug,
  kind,
  instructions,
  name,
  onText,
  signal,
}: ImproveInstructionsRequest): Promise<string> {
  const res = await fetch(
    `/api/${encodeURIComponent(orgSlug)}/instructions/improve`,
    {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, instructions, name }),
      signal,
    },
  );
  if (!res.ok || !res.body) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error || `Request failed (HTTP ${res.status})`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += value;
    onText(text);
  }
  // A text stream carries no error frame: a model failure ends it empty.
  const final = text.trim();
  if (!final) throw new Error("The model returned no text");
  return final;
}
