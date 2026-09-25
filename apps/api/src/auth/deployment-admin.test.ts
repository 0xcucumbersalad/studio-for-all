import { beforeEach, describe, expect, mock, test } from "bun:test";
import {
  deploymentAdminVerdict,
  resetFirstUserCacheForTest,
} from "./deployment-admin";

const settings = (over: {
  emails?: string[];
  firstUser?: boolean;
  plans?: boolean;
}) => ({
  deploymentAdminEmails: over.emails ?? [],
  deploymentAdminFirstUser: over.firstUser ?? true,
  plansEnabled: over.plans ?? false,
});

const first = "user_first";
const lookupFirstUser = () => Promise.resolve(first);

describe("deploymentAdminVerdict", () => {
  beforeEach(() => resetFirstUserCacheForTest());

  test("an allowlisted, verified email is admin", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: "u", email: "Ops@Example.com", emailVerified: true },
        { settings: settings({ emails: ["ops@example.com"] }) },
      ),
    ).toBe("admin");
  });

  test("an allowlisted but unverified email is told to verify", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: "u", email: "ops@example.com", emailVerified: false },
        { settings: settings({ emails: ["ops@example.com"] }) },
      ),
    ).toBe("unverified");
  });

  test("a set allowlist disables the first-account rule", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: first, email: "someone@example.com", emailVerified: true },
        {
          settings: settings({ emails: ["ops@example.com"] }),
          lookupFirstUser,
        },
      ),
    ).toBe("denied");
  });

  test("with no allowlist, the first account is admin, verified or not", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: first, email: "a@example.com", emailVerified: false },
        { settings: settings({}), lookupFirstUser },
      ),
    ).toBe("admin");
  });

  test("with no allowlist, every other account is denied", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: "user_second", email: "b@example.com", emailVerified: true },
        { settings: settings({}), lookupFirstUser },
      ),
    ).toBe("denied");
  });

  test("DEPLOYMENT_ADMIN_FIRST_USER=false turns the rule off", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: first },
        { settings: settings({ firstUser: false }), lookupFirstUser },
      ),
    ).toBe("denied");
  });

  test("never applies on a deployment with plans enabled", async () => {
    expect(
      await deploymentAdminVerdict(
        { id: first },
        { settings: settings({ plans: true }), lookupFirstUser },
      ),
    ).toBe("denied");
  });

  test("an empty user table is not cached", async () => {
    const lookup = mock<() => Promise<string | null>>(() =>
      Promise.resolve(null),
    );
    const deps = { settings: settings({}), lookupFirstUser: lookup, now: 0 };
    expect(await deploymentAdminVerdict({ id: first }, deps)).toBe("denied");
    lookup.mockImplementation(() => Promise.resolve(first));
    expect(await deploymentAdminVerdict({ id: first }, deps)).toBe("admin");
  });

  test("the first account is cached, then re-read after the TTL", async () => {
    const lookup = mock(() => Promise.resolve(first));
    const s = settings({});
    await deploymentAdminVerdict(
      { id: first },
      { settings: s, lookupFirstUser: lookup, now: 0 },
    );
    await deploymentAdminVerdict(
      { id: first },
      { settings: s, lookupFirstUser: lookup, now: 60_000 },
    );
    expect(lookup).toHaveBeenCalledTimes(1);
    await deploymentAdminVerdict(
      { id: first },
      { settings: s, lookupFirstUser: lookup, now: 6 * 60_000 },
    );
    expect(lookup).toHaveBeenCalledTimes(2);
  });
});
