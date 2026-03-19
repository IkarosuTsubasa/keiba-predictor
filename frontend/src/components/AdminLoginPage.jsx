import React, { useEffect, useMemo, useState } from "react";

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
  const legacyConsoleUrl = useMemo(() => {
    const trimmed = String(token || "").trim();
    if (!trimmed) {
      return `${appBasePath}/console`;
    }
    return `${appBasePath}/console?token=${encodeURIComponent(trimmed)}`;
  }, [appBasePath, token]);

  useEffect(() => {
    document.title = "管理员登录";
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
          setError("保存的管理员口令已失效，请重新登录。");
        }
      })
      .catch((fetchError) => {
        setError(fetchError?.message || "管理员验证失败。");
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

    const url = trimmed ? `${authCheckUrl}?token=${encodeURIComponent(trimmed)}` : authCheckUrl;
    fetch(url, { headers })
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
          setError("管理员口令无效。");
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
          window.location.href = trimmed ? `${appBasePath}/console?token=${encodeURIComponent(trimmed)}` : `${appBasePath}/console`;
        }
      })
      .catch((fetchError) => {
        setReady(false);
        setError(fetchError?.message || "管理员验证失败。");
      })
      .finally(() => {
        setChecking(false);
      });
  }

  return (
    <main className="admin-login-page">
      <section className="admin-login-card">
        <span className="admin-login-card__eyebrow">Admin Console</span>
        <h1>管理员登录</h1>
        <p>
          这里先作为 React 入口页使用。登录成功后会继续跳转到当前仍在使用的旧控制台，后续再逐步迁到
          React 管理端。
        </p>

        <form className="admin-login-form" onSubmit={handleSubmit}>
          <label className="admin-login-form__field">
            <span>管理员口令</span>
            <input
              type="password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder={enabled ? "ADMIN_TOKEN" : "当前未启用 ADMIN_TOKEN"}
            />
          </label>

          {error ? <div className="admin-login-form__error">{error}</div> : null}

          <div className="admin-login-form__actions">
            <button type="submit" disabled={checking}>
              {checking ? "验证中..." : "进入控制台"}
            </button>
            {ready ? (
              <a href={legacyConsoleUrl} className="admin-login-form__link">
                打开旧控制台
              </a>
            ) : (
              <a href={appBasePath} className="admin-login-form__link admin-login-form__link--ghost">
                返回公开页
              </a>
            )}
          </div>
        </form>
      </section>
    </main>
  );
}

export { ADMIN_TOKEN_STORAGE_KEY };
