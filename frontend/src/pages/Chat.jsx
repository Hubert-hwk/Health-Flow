import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Input,
  List,
  message,
  Space,
  Spin,
  Tag,
  Typography,
} from 'antd';
import {
  SendOutlined,
  AimOutlined,
  SafetyCertificateOutlined,
  RobotOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { chat, chatStream, routeQuery } from '../api.js';

const { TextArea } = Input;

function pickTopIntent(distribution) {
  if (!distribution || typeof distribution !== 'object') return null;
  return Object.entries(distribution).sort((a, b) => b[1] - a[1])[0] || null;
}

function getRedFlag(safety) {
  if (!safety || typeof safety !== 'object') return false;
  if (safety.red_flag !== undefined) return Boolean(safety.red_flag);
  if (safety['红旗标记'] !== undefined) return Boolean(safety['红旗标记']);
  return false;
}

export default function Chat() {
  const [input, setInput] = useState('');
  const [patientId, setPatientId] = useState('');
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [sending, setSending] = useState(false);

  const [routing, setRouting] = useState(null);
  const [routingLoading, setRoutingLoading] = useState(false);

  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput('');

    const body = { message: text, include_history: true };
    if (sessionId) body.session_id = sessionId;
    if (patientId.trim()) body.patient_id = patientId.trim();

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
      await chatStream(body, {
        onDelta: (c) => updateLast((m) => ({ ...m, content: m.content + c })),
        onRoute: (evt) => updateLast((m) => ({ ...m, route: evt })),
        onDone: (evt) => {
          updateLast((m) => ({ ...m, streaming: false, meta: evt }));
          if (evt && evt.session_id) setSessionId(evt.session_id);
        },
        onError: (err) => { throw err; },
      });
    } catch (err) {
      try {
        const data = await chat(body);
        updateLast((m) => ({
          ...m,
          streaming: false,
          meta: data,
          content: (data && data.reply) || m.content,
        }));
        if (data && data.session_id) setSessionId(data.session_id);
      } catch (err2) {
        updateLast((m) => ({ ...m, streaming: false, error: err2.message }));
      }
    } finally {
      setSending(false);
    }
  };

  const doRouting = async () => {
    const text = input.trim();
    if (!text) { message.warning('请先输入要分诊的问题'); return; }
    setRoutingLoading(true);
    try {
      const body = { query: text };
      if (patientId.trim()) body.patient_id = patientId.trim();
      const data = await routeQuery(body);
      setRouting(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setRoutingLoading(false);
    }
  };

  const top = routing ? pickTopIntent(routing.intent_distribution) : null;

  return (
    <div className="page-stack">
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space wrap>
          <Input
            placeholder="患者编号（可选，例如 P001）"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            style={{ width: 220 }}
            allowClear
          />
          {sessionId && <Tag color="blue">会话 ID：{sessionId.slice(0, 12)}…</Tag>}
          <Typography.Text type="secondary">会话在内存中保存，刷新后重置</Typography.Text>
        </Space>
      </Card>

      <Card
        title={<Space><RobotOutlined /> 健康流智能助手</Space>}
        extra={
          <Space>
            <Button icon={<AimOutlined />} loading={routingLoading} onClick={doRouting}>仅分诊</Button>
            <Button type="primary" icon={<SendOutlined />} loading={sending} onClick={send}>发送</Button>
          </Space>
        }
      >
        <div className="chat-window">
          {messages.length === 0 && (
            <div className="chat-empty">
              <RobotOutlined style={{ fontSize: 40, color: '#1677ff' }} />
              <Typography.Text type="secondary">
                你好！我是健康流智能助手，可以帮你解读体检报告、分诊科室、解答健康问题。
              </Typography.Text>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className={`chat-avatar ${m.role}`}>
                {m.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
              </div>
              <div className="chat-body">
                <div className={`chat-bubble ${m.role}`}>
                  {m.role === 'assistant' && m.streaming && (
                    <Spin size="small" style={{ marginRight: 8 }} />
                  )}
                  {m.content || (m.streaming ? '正在思考…' : '')}
                </div>
                {m.error && <Alert style={{ marginTop: 6 }} type="error" showIcon title={m.error} />}
                {m.role === 'assistant' && !m.streaming && !m.error && m.meta && (
                  <ChatMeta meta={m.meta} />
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入你的健康问题，Enter 发送，Shift+Enter 换行"
          autoSize={{ minRows: 2, maxRows: 6 }}
          style={{ marginTop: 16 }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
      </Card>
    </div>
  );
}

function ChatMeta({ meta }) {
  const safety = meta.safety_check || meta.safetyCheck;
  const red = getRedFlag(safety);
  const agent = meta.agent_used || meta.agent;
  const dept = meta.department || (meta.route && meta.route.routed_department);
  const topIntent = pickTopIntent(meta.intent_distribution);
  const refs = Array.isArray(meta.references) ? meta.references : [];

  return (
    <div className="chat-meta">
      <Space wrap size={[6, 6]} style={{ marginBottom: 6 }}>
        {dept && <Tag color="geekblue">分诊科室：{dept}</Tag>}
        {agent && <Tag color="purple">Agent：{agent}</Tag>}
        {topIntent && <Tag color="cyan">置信度：{topIntent[0]} {Math.round(topIntent[1] * 100)}%</Tag>}
        {safety && (
          red
            ? <Tag color="red" icon={<SafetyCertificateOutlined />}>安全校验：🚩 红旗</Tag>
            : <Tag color="green" icon={<SafetyCertificateOutlined />}>安全校验：✓ 通过</Tag>
        )}
      </Space>
      {safety && Array.isArray(safety.warnings) && safety.warnings.length > 0 && (
        <Typography.Paragraph type="warning" style={{ marginBottom: 6 }}>
          安全警告：{safety.warnings.join('；')}
        </Typography.Paragraph>
      )}
      {refs.length > 0 && (
        <List
          size="small"
          header={<Typography.Text strong>参考证据</Typography.Text>}
          dataSource={refs}
          renderItem={(r) => (
            <List.Item>
              <Space direction="vertical" size={0}>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  [{r.type}] {r.source_id || '—'} {r.score != null ? `得分 ${Number(r.score).toFixed(3)}` : ''}
                </Typography.Text>
                <Typography.Text style={{ fontSize: 12 }}>{r.content || '—'}</Typography.Text>
              </Space>
            </List.Item>
          )}
        />
      )}
    </div>
  );
}
