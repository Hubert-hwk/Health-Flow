// api.js —— 所有后端请求的集中封装。
// 统一使用相对路径 /api/...，由 Vite 开发服务器代理到 http://localhost:8080，
// 从而避免跨域问题，并保证生产部署时可通过反向代理转发。

const BASE = '/api';

/**
 * 从非 2xx 响应中提取后端返回的错误信息（优先 detail 字段）。
 */
export async function parseError(res) {
  let detail = `请求失败（HTTP ${res.status}）`;
  try {
    const data = await res.json();
    if (data && typeof data.detail === 'string') {
      detail = data.detail;
    } else if (data && data.detail !== undefined) {
      // detail 可能是数组（如 422 校验错误），转成可读文本
      detail = JSON.stringify(data.detail);
    } else if (data && typeof data.message === 'string') {
      detail = data.message;
    }
  } catch (e) {
    // 响应体不是 JSON，保留默认错误信息
  }
  return detail;
}

/**
 * 通用请求封装：非 2xx 一律抛出带中文 detail 的 Error。
 */
async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  if (res.status === 204) return null;
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) return res.json();
  return res.text();
}

// 不拼 /api 前缀的原始请求（/health、/ready 挂在后端根路径）
async function rawRequest(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  if (res.status === 204) return null;
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) return res.json();
  return res.text();
}

function jsonOptions(method, body) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

/* ---------------- 系统状态 ---------------- */

// GET /health —— 后端健康状态（根路径，不带 /api 前缀）
export const getHealth = () => rawRequest('/health');

// GET /ready —— 后端就绪状态（根路径，不带 /api 前缀；数据库 / Milvus / Neo4j 等依赖）
export const getReady = () => rawRequest('/ready');

/* ---------------- 报告 ---------------- */

// GET /api/health/reports —— 报告列表
export function getReports({ patient_id, department, limit, offset } = {}) {
  const params = new URLSearchParams();
  if (patient_id) params.set('patient_id', patient_id);
  if (department) params.set('department', department);
  if (limit) params.set('limit', limit);
  if (offset) params.set('offset', offset);
  const qs = params.toString();
  return request(`/health/reports${qs ? `?${qs}` : ''}`);
}

// POST /api/health/report/upload —— 上传报告（multipart 表单）
export function uploadReport(formData) {
  return request('/health/report/upload', { method: 'POST', body: formData });
}

// GET /api/health/report/{id} —— 报告详情（含指标）
export const getReport = (id) => request(`/health/report/${id}`);

// GET /api/health/report/{id}/metrics —— 报告指标列表
export const getReportMetrics = (id) => request(`/health/report/${id}/metrics`);

// DELETE /api/health/report/{report_id} —— 删除报告
export const deleteReport = (reportId) =>
  request(`/health/report/${reportId}`, { method: 'DELETE' });

/* ---------------- 指标分析 ---------------- */

// GET /api/health/metric/trend —— 指标趋势
export function getMetricTrend({ patient_id, metric_name, days } = {}) {
  const params = new URLSearchParams();
  if (patient_id) params.set('patient_id', patient_id);
  if (metric_name) params.set('metric_name', metric_name);
  if (days) params.set('days', days);
  return request(`/health/metric/trend?${params.toString()}`);
}

// GET /api/health/metric/anomalies —— 异常指标汇总
export function getMetricAnomalies({ patient_id, days } = {}) {
  const params = new URLSearchParams();
  if (patient_id) params.set('patient_id', patient_id);
  if (days) params.set('days', days);
  return request(`/health/metric/anomalies?${params.toString()}`);
}

// GET /api/health/metric/search —— 指标搜索（该接口存在但未在契约列表中）
export function getMetricSearch({ patient_id, keyword, abnormal_only, limit } = {}) {
  const params = new URLSearchParams();
  if (patient_id) params.set('patient_id', patient_id);
  if (keyword) params.set('keyword', keyword);
  if (abnormal_only) params.set('abnormal_only', abnormal_only ? 'true' : 'false');
  if (limit) params.set('limit', limit);
  return request(`/health/metric/search?${params.toString()}`);
}

/* ---------------- 智能问答 ---------------- */

// POST /api/health/chat —— 非流式问答
export const chat = (body) => request('/health/chat', jsonOptions('POST', body));

// POST /api/health/chat/stream —— 流式问答（SSE）
// 回调：onRoute(routeEvent), onDelta(content), onDone(doneEvent), onError(err)
export async function chatStream(body, { onRoute, onDelta, onDone, onError } = {}) {
  const res = await fetch(`${BASE}/health/chat/stream`, jsonOptions('POST', body));
  if (!res.ok) throw new Error(await parseError(res));
  if (!res.body) throw new Error('后端未返回流式响应体');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // 逐块读取 SSE 文本，按行解析 "data: {...}" 事件
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // 最后一行可能不完整，留到下一轮
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue; // 忽略注释/事件名等
      const payload = trimmed.slice(5).trim();
      if (!payload) continue;
      let evt;
      try {
        evt = JSON.parse(payload);
      } catch (e) {
        continue; // 无法解析的行直接跳过
      }
      if (evt.type === 'delta' && onDelta) onDelta(evt.content ?? '');
      else if (evt.type === 'route' && onRoute) onRoute(evt);
      else if (evt.type === 'done' && onDone) onDone(evt);
      else if (evt.type === 'error' && onError) onError(new Error(evt.message || evt.content || '流式响应出错'));
    }
  }
}

// POST /api/health/routing —— 仅分诊（不生成回答）
export const routeQuery = (body) => request('/health/routing', jsonOptions('POST', body));

// GET /api/health/safety/check —— 安全校验（单独调用）
export function safetyCheck(content) {
  const params = new URLSearchParams({ content: content || '' });
  return request(`/health/safety/check?${params.toString()}`);
}

/* ---------------- 知识图谱 ---------------- */

// GET /api/health/kg/department/{symptom} —— 症状到科室的图谱查询
export const kgDepartment = (symptom) =>
  request(`/health/kg/department/${encodeURIComponent(symptom)}`);
