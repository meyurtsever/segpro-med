import { useCallback, useState } from 'react';

import * as api from '../api/client';

interface CrowdsourcingLoginModalProps {
  onClose: () => void;
  onSuccess: (session: api.LoginResponse) => void;
}

const backdropStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 1000,
  background: 'rgba(0, 0, 0, 0.46)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 20,
};

const modalStyle: React.CSSProperties = {
  width: 390,
  maxWidth: '100%',
  borderRadius: 8,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-secondary)',
  boxShadow: 'var(--shadow)',
  overflow: 'hidden',
};

const headerStyle: React.CSSProperties = {
  padding: '16px 18px 12px',
  borderBottom: '1px solid var(--border-color)',
};

const bodyStyle: React.CSSProperties = {
  padding: 18,
};

const labelStyle: React.CSSProperties = {
  display: 'block',
  color: 'var(--text-secondary)',
  fontSize: 11,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: 0,
  marginBottom: 5,
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 9px',
  borderRadius: 6,
  border: '1px solid var(--border-color)',
  background: 'var(--bg-tertiary)',
  color: 'var(--text-primary)',
  outline: 'none',
  boxSizing: 'border-box',
  fontSize: 13,
};

const buttonStyle = (color: string, disabled = false): React.CSSProperties => ({
  padding: '8px 11px',
  borderRadius: 6,
  border: `1px solid color-mix(in srgb, ${color} 55%, var(--border-color))`,
  background: disabled
    ? 'var(--bg-tertiary)'
    : `color-mix(in srgb, ${color} 15%, var(--bg-secondary))`,
  color,
  fontSize: 12,
  fontWeight: 800,
  cursor: disabled ? 'not-allowed' : 'pointer',
  opacity: disabled ? 0.55 : 1,
});

const demoCredentialStyle: React.CSSProperties = {
  border: '1px solid color-mix(in srgb, var(--accent-blue) 28%, var(--border-color))',
  background: 'color-mix(in srgb, var(--accent-blue) 7%, var(--bg-secondary))',
  borderRadius: 7,
  padding: '9px 10px',
  color: 'var(--text-secondary)',
  fontSize: 12,
  lineHeight: 1.5,
  marginBottom: 12,
};

export default function CrowdsourcingLoginModal({
  onClose,
  onSuccess,
}: CrowdsourcingLoginModalProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async () => {
    if (!username.trim() || !password) {
      setError('Username and password are required.');
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const session = await api.login(username, password);
      onSuccess(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed.');
    } finally {
      setBusy(false);
    }
  }, [onSuccess, password, username]);

  return (
    <div style={backdropStyle} role="presentation" onMouseDown={onClose}>
      <div
        style={modalStyle}
        role="dialog"
        aria-modal="true"
        aria-labelledby="crowdsourcing-login-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div style={headerStyle}>
          <div
            id="crowdsourcing-login-title"
            style={{ color: 'var(--text-primary)', fontSize: 16, fontWeight: 900 }}
          >
            Crowdsourcing Login
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 5, lineHeight: 1.45 }}>
            Uses the same `db/users.txt` credentials and assignment store as the Gradio app.
          </div>
        </div>

        <div style={bodyStyle}>
          <div style={{ marginBottom: 12 }}>
            <label style={labelStyle}>Username</label>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void submit();
                }
              }}
              autoFocus
              style={inputStyle}
            />
          </div>

          <div style={{ marginBottom: 12 }}>
            <label style={labelStyle}>Password</label>
            <input
              value={password}
              type="password"
              onChange={(event) => setPassword(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void submit();
                }
              }}
              style={inputStyle}
            />
          </div>

          {error ? (
            <div
              style={{
                padding: '8px 9px',
                borderRadius: 6,
                border: '1px solid rgba(224, 92, 92, 0.28)',
                background: 'rgba(224, 92, 92, 0.08)',
                color: 'var(--accent-red)',
                fontSize: 12,
                lineHeight: 1.4,
                marginBottom: 12,
              }}
            >
              {error}
            </div>
          ) : null}

          <div style={demoCredentialStyle}>
            <div style={{ color: 'var(--accent-blue)', fontSize: 11, fontWeight: 900, textTransform: 'uppercase' }}>
              Demo Credentials
            </div>
            <div style={{ marginTop: 4 }}>
              Admin: <code>admin1</code> / <code>adminpass</code>
            </div>
            <div>
              Expert: <code>john_doe</code> / <code>pass123</code>
            </div>
            <div>
              Expert: <code>jane_smith</code> / <code>expert456</code>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button type="button" onClick={onClose} style={buttonStyle('var(--text-secondary)')}>
              Cancel
            </button>
            <button
              type="button"
              onClick={() => { void submit(); }}
              disabled={busy}
              style={buttonStyle('var(--accent-green)', busy)}
            >
              {busy ? 'Signing in...' : 'Sign In'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
