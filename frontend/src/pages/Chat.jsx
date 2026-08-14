import React, { useEffect, useRef, useState } from 'react';
import { chat, chatStream, routeQuery } from '../api.js';

// 生成一个内存中的会话 ID（会话仅保存在当前页面内存中）
function makeSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// 从意图分布中提取置信度最高的科室及置信度
function pickTopIntent(dist) {
  if (!dist || typeof dist !== 'object') return null;
  let best = null;
  for (const [k, v] of Object.entries(dist)) {
    const score = Number(v);
    if (Number.isFinite(score) && (!best || score > best.score)) best = { dept: k, score };
  }
  return best;
}

// 读取安全校验中的红旗标记：后端键名为中文「红旗标记」，兼容 red_flag
function getRedFlag(safety) {
  if (!safety || typeof safety !== 'object') return false;
  if (safety['红旗标记'] !== undefined) return Boolean(safety['红旗标记']);
  if (safety.red_flag !== undefined) return Boolean(safety.red_flag);
  return false;
}

// 安全校验徽章
function SafetyBadge({ safety }) {
  if (!safety) return null;
  const red = getRedFlag(safety);
  const passed = safety.passed !== false;
  return (
    <span className={`badge ${red ? 'badge-red' : passed ? 'badge-normal' : 'badge-warn'}`}>
      {red ? '🚩 红旗' : passed ? '✓ 通过' : '⚠ 未通过'}
    </span>
  );
}

// 助手消息的元信息面板（科室 / Agent / 置信度 / 参考 / 安全 / 反馈）
function MetaPanel({ meta, route }) {
  const dept = meta?.department || route?.department || route?.routed_department;
  const agent = meta?.agent_used || route?.agent_used;
  const top = pickTopIntent(meta?.intent_distribution || route?.intent_distribution);
  const references = meta?.references || [];
  const safety = meta?.safety_check;
  const feedback = meta?.feedback_info;

  return (
    <details className="meta-panel" open>
      <summary>详细信息（科室 / 参考来源 / 安全校验）</summary>
      <div className="meta-grid">
        <div className="meta-item">
          <span className="meta-label">分诊科室</span>
          <span className="meta-value">{dept || '—'}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">使用 Agent</span>
          <span className="meta-value">{agent || '—'}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">置信度</span>
          <span className="meta-value">
            {top ? `${top.dept} ${(top.score * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">安全校验</span>
          <span className="meta-value"><SafetyBadge safety={safety} /></span>
        </div>
      </div>

      {safety && (
        <div className="meta-section">
          <div className="meta-label">安全校验详情</div>
          <p className="hint">
            {safety.warnings && safety.warnings.length > 0
              ? `警告：${safety.warnings.join('；')}`
              : '无警告'}
            {safety.critical ? ' ｜ 严重：是' : ''}
          </p>
        </div>
      )}

      {feedback && (
        <div className="meta-section">
          <div className="meta-label">反馈信息</div>
          <p className="hint">
            一致性检查：{feedback.consistency_check ?? '—'}
            ｜ 证据评分：{feedback.evidence_score ?? '—'}
            ｜ 矛盾数：{feedback.contradictions ?? '—'}
            ｜ 递归深度：{feedback.recursion_depth ?? '—'}
          </p>
        </div>
      )}

      {references.length > 0 && (
        <div className="meta-section">
          <div className="meta-label">参考来源（{references.length}）</div>
          <ul className="ref-list">
            {references.map((r, i) => (
              <li key={i}>
                <span className="ref-type">{r.type || '来源'}</span>
                {r.source_id !== undefined && <span className="ref-src">#{r.source_id}</span>}
                <span className="ref-content">{r.content || '—'}</span>
                {r.score !== undefined && <span className="ref-score">得分 {Number(r.score).toFixed(2)}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </details>
  );
}

export default function Chat() {
  const [patientId, setPatientId] = useState('');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sending, setSending] = useState(false);

  // 仅分诊结果
  const [routing, setRouting] = useState(null);
  const [routingLoading, setRoutingLoading] = useState(false);
  const [routingError, setRoutingError] = useState('');

  const bottomRef = useRef(null);

  // 新消息时自动滚动到底部
  useEffect(() => {
    bottomRef.current && bottomRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput('');

    // 首次发送时生成会话 ID，保存在内存中以支持多轮对话
    const sid = sessionId || makeSessionId();
    if (!sessionId) setSessionId(sid);

    const body = { message: text, include_history: true, session_id: sid };
    if (patientId.trim()) body.patient_id = patientId.trim();

    // 先插入用户消息与空的助手消息（流式输出时逐步填充）
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '', streaming: true, meta: null, error: null },
    ]);
    setSending(true);

    const updateLast = (fn) =>
      setMessages((prev) => {
        const next = [...prev];
        const idx = next.length - 1;
        if (idx >= 0 && next[idx] && next[idx].streaming) next[idx] = fn(next[idx]);
        return next;
      });

    try {
      // 优先使用 SSE 流式接口，失败时回退到非流式接口
      await chatStream(body, {
        onDelta: (c) => updateLast((m) => ({ ...m, content: m.content + c })),
        onRoute: (evt) => updateLast((m) => ({ ...m, route: evt })),
        onDone: (evt) => updateLast((m) => ({ ...m, streaming: false, meta: evt })),
        onError: (err) => { throw err; },
      });
    } catch (err) {
      // 流式不可用 → 回退 POST /api/health/chat
      try {
        const data = await chat(body);
        updateLast((m) => ({
          ...m,
          streaming: false,
          meta: data,
          content: (data && data.reply) || m.content,
        }));
      } catch (err2) {
        updateLast((m) => ({ ...m, streaming: false, error: err2.message }));
      }
    } finally {
      setSending(false);
    }
  };

  const doRouting = async () => {
    const text = input.trim();
    if (!text) { setRoutingError('请先输入要分诊的问题'); return; }
    setRoutingError('');
    setRoutingLoading(true);
    try {
      const body = { query: text };
      if (patientId.trim()) body.patient_id = patientId.trim();
      const data = await routeQuery(body);
      setRouting(data);
    } catch (err) {
      setRoutingError(err.message);
      setRouting(null);
    } finally {
      setRoutingLoading(false);
    }
  };

  const top = routing ? pickTopIntent(routing.intent_distribution) : null;

  return (
    <div>
      {/* 患者编号（可选，用于上下文） */}
      <div className="card">
        <div className="form-inline">
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="患者编号（可选，例如 P001）"
          />
          {sessionId && <span className="hint">会话 ID：{sessionId.slice(0, 8)}…（内存中保存，刷新后重置）</span>}
        </div>
      </div>

      {/* 聊天窗口 */}
      <div className="card chat-card">
        <div className="chat-window">
          {messages.length === 0 && (
            <p className="muted chat-placeholder">
              你好！我是健康流智能助手，可以帮你解读体检报告、分诊科室、解答健康问题。
            </p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="chat-bubble">
                {m.role === 'assistant' && m.streaming && (
                  <span className="typing">正在思考<span className="dots">…</span></span>
                )}
                {m.content || (m.streaming ? '' : '')}
                {m.error && <div className="alert alert-error">{m.error}</div>}
              </div>
              {m.role === 'assistant' && !m.streaming && !m.error && (
                <MetaPanel meta={m.meta} route={m.route} />
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="输入你的健康问题，Enter 发送，Shift+Enter 换行"
            rows={3}
          />
          <div className="chat-actions">
            <button className="btn btn-secondary" onClick={doRouting} disabled={routingLoading || sending}>
              {routingLoading ? '分诊中…' : '仅分诊'}
            </button>
            <button className="btn btn-primary" onClick={send} disabled={sending || !input.trim()}>
              {sending ? '发送中…' : '发送'}
            </button>
          </div>
        </div>
      </div>

      {/* 仅分诊结果 */}
      {routingError && <div className="alert alert-error">{routingError}</div>}
      {routing && (
        <div className="card">
          <h3 className="card-title">分诊结果</h3>
          <div className="meta-grid">
            <div className="meta-item">
              <span className="meta-label">目标科室</span>
              <span className="meta-value">{routing.routed_department || '—'}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">置信度</span>
              <span className="meta-value">
                {routing.confidence !== undefined
                  ? `${(Number(routing.confidence) * 100).toFixed(1)}%`
                  : top ? `${top.dept} ${(top.score * 100).toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="meta-item">
              <span className="meta-label">低置信度</span>
              <span className="meta-value">{routing.low_confidence ? '是' : '否'}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">需人工复核</span>
              <span className="meta-value">{routing.human_review_required ? '是' : '否'}</span>
            </div>
          </div>
          {routing.intent_distribution && (
            <div className="meta-section">
              <div className="meta-label">意图分布</div>
              <p className="hint">
                {Object.entries(routing.intent_distribution)
                  .map(([k, v]) => `${k}: ${(Number(v) * 100).toFixed(1)}%`)
                  .join(' ｜ ')}
              </p>
            </div>
          )}
          {routing.reasoning && (
            <div className="meta-section">
              <div className="meta-label">推理过程</div>
              <p className="hint">{routing.reasoning}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
