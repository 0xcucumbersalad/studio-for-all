/**
 * Who is a deployment admin — the one answer the `/api/_admin/*` fence and the
 * GitHub App manifest callback share.
 *
 * 1. `DEPLOYMENT_ADMIN_EMAILS` lists them. A listed email must be verified:
 *    otherwise anyone could sign up with the operator's address and walk in.
 * 2. With that list EMPTY, the deployment's first account is the admin — so a
 *    fresh self-hosted install can reach the admin dashboard (and set GitHub
 *    up in one click) without editing its environment first. Keyed on the
 *    account, not its email, so no verification is needed: nobody else can
 *    become the oldest account after the fact. Off when
 *    `DEPLOYMENT_ADMIN_FIRST_USER=false`, and never on a deployment with plans
 *    enabled — a hosted multi-tenant instance's oldest account belongs to a
 *    customer, not an operator.
 *
 * The first-account rule's trade-off is the usual one for self-hosted tools: on
 * an instance reachable before its operator signs up, whoever signs up first
 * owns it. Operators who expose the URL early set the list instead.
 */

import { getDb } from "@/database";
import { getSettings } from "@/settings";

export type DeploymentAdminVerdict = "admin" | "unverified" | "denied";

interface AdminCandidate {
  id: string;
  email?: string | null;
  emailVerified?: boolean | null;
}

const FIRST_USER_TTL_MS = 5 * 60_000;
let firstUser: { id: string | null; at: number } | null = null;

/** The oldest account's id. Cached briefly: it only changes on deletion. */
async function firstUserId(
  lookup: () => Promise<string | null>,
  now: number,
): Promise<string | null> {
  if (firstUser && now - firstUser.at < FIRST_USER_TTL_MS) return firstUser.id;
  const id = await lookup();
  // Only a found id is cached: an empty table must not pin "nobody" for five
  // minutes while the operator is signing up.
  firstUser = id ? { id, at: now } : null;
  return id;
}

async function lookupFirstUserId(): Promise<string | null> {
  const row = await getDb()
    .db.selectFrom("user")
    .select("id")
    .orderBy("createdAt", "asc")
    .orderBy("id", "asc")
    .limit(1)
    .executeTakeFirst();
  return row?.id ?? null;
}

export async function deploymentAdminVerdict(
  user: AdminCandidate,
  deps: {
    settings?: Pick<
      ReturnType<typeof getSettings>,
      "deploymentAdminEmails" | "deploymentAdminFirstUser" | "plansEnabled"
    >;
    lookupFirstUser?: () => Promise<string | null>;
    now?: number;
  } = {},
): Promise<DeploymentAdminVerdict> {
  const settings = deps.settings ?? getSettings();
  if (settings.deploymentAdminEmails.length > 0) {
    const email = user.email?.toLowerCase();
    if (!email || !settings.deploymentAdminEmails.includes(email)) {
      return "denied";
    }
    return user.emailVerified ? "admin" : "unverified";
  }
  if (!settings.deploymentAdminFirstUser || settings.plansEnabled) {
    return "denied";
  }
  const first = await firstUserId(
    deps.lookupFirstUser ?? lookupFirstUserId,
    deps.now ?? Date.now(),
  );
  return first !== null && first === user.id ? "admin" : "denied";
}

/** Test seam: forget the cached first account. */
export function resetFirstUserCacheForTest(): void {
  firstUser = null;
}
