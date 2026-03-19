import React, { useEffect, useState } from "react";

const ADMIN_TOKEN_STORAGE_KEY = "ikaimo_admin_token";

export default function AdminLoginPage({
  appBasePath = "/keiba",
  redirectToLegacy = true,
  onAuthenticated = null,
}) {
  const [token, setToken] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [enabled, setEnabled] = useState(true);

  const authCheckUrl = `${appBasePath}/api/admin/auth-check`;

  useEffect(() => {
    document.title = "Admin Console";
  }, []);

  useEffect(() => {
    const stored = window.sessionStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || "";
    if (!stored.trim()) {
      return;
    }
    setToken(stored);
    setChecking(true);
    fetch(authCheckUrl, {
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${stored}`,
      },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        const valid = Boolean(data?.valid);
        setEnabled(Boolean(data?.enabled));
        setReady(valid);
        if (!valid) {
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          setError("Admin token invalid.");
        }
      })
      .catch((fetchError) => {
        setError(fetchError?.message || "Auth check failed.");
      })
      .finally(() => {
        setChecking(false);
      });
  }, [authCheckUrl]);

  function handleSubmit(event) {
    event.preventDefault();
    const trimmed = String(token || "").trim();
    setChecking(true);
    setError("");

    const headers = { Accept: "application/json" };
    if (trimmed) {
      headers.Authorization = `Bearer ${trimmed}`;
    }

    fetch(authCheckUrl, { headers })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        const valid = Boolean(data?.valid);
        setEnabled(Boolean(data?.enabled));
        if (!valid) {
          setReady(false);
          setError("Admin token invalid.");
          window.sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
          return;
        }
        window.sessionStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, trimmed);
        setReady(true);
        if (typeof onAuthenticated === "function") {
          onAuthenticated(trimmed);
          return;
        }
        if (redirectToLegacy) {
          window.location.href = `${appBasePath}/console`;
        }
      })
      .catch((fetchError) => {
        setReady(false);
        setError(fetchError?.message || "Auth check failed.");
      })
      .finally(() => {
        setChecking(false);
      });
  }

  return (
    <main className="admin-login-page">
      <section className="admin-login-card">
        <span className="admin-login-card__eyebrow">Admin Console</span>
        <h1>Admin Login</h1>
        <p>Use the admin token to open the React control panel.</p>

        <form className="admin-login-form" onSubmit={handleSubmit}>
          <label className="admin-login-form__field">
            <span>Admin Token</span>
            <input
              type="password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder={enabled ? "ADMIN_TOKEN" : "ADMIN_TOKEN disabled"}
            />
          </label>

          {error ? <div className="admin-login-form__error">{error}</div> : null}

          <div className="admin-login-form__actions">
            <button type="submit" disabled={checking}>
              {checking ? "Checking..." : "Sign In"}
            </button>
            {ready ? (
              <a href={`${appBasePath}/console`} className="admin-login-form__link">
                Open Console
              </a>
            ) : (
              <a href={appBasePath} className="admin-login-form__link admin-login-form__link--ghost">
                Back to Public
              </a>
            )}
          </div>
        </form>
      </section>
    </main>
  );
}

export { ADMIN_TOKEN_STORAGE_KEY };
