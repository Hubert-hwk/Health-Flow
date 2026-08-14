import React, { useState } from 'react';
import { getReports, getReport, getReportMetrics, deleteReport } from '../api.js';

// 报告列表页：按患者编号查询报告，点击查看详情，支持删除
export default function Reports() {
  const [patientId, setPatientId] = useState('');
  const [reports, setReports] = useState(null);   // null 表示尚未查询
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null); // 当前查看的报告详情
  const [detailMetrics, setDetailMetrics] = useState([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');
  const [deletingId, setDeletingId] = useState(null);

  const search = async (e) => {
    if (e) e.preventDefault();
    if (!patientId.trim()) {
      setError('请填写患者编号');
      return;
    }
    setError('');
    setSelected(null);
    setLoading(true);
    try {
      const data = await getReports({ patient_id: patientId.trim() });
      setReports(Array.isArray(data) ? data : (data && data.reports) || []);
    } catch (err) {
      setError(err.message);
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  // 点击某条报告：并行加载详情与指标
  const openDetail = async (report) => {
    setSelected(report);
    setDetailMetrics([]);
    setDetailError('');
    setDetailLoading(true);
    try {
      const [detail, metrics] = await Promise.all([
        getReport(report.id),
        getReportMetrics(report.id),
      ]);
      // 详情对象优先，指标列表兜底
      setSelected(detail && detail.id !== undefined ? detail : report);
      const ms = Array.isArray(metrics) ? metrics : (detail && detail.metrics) || [];
      setDetailMetrics(ms);
    } catch (err) {
      setDetailError(err.message);
      // 详情接口失败时，尝试只用指标接口的结果
      try {
        const metrics = await getReportMetrics(report.id);
        setDetailMetrics(Array.isArray(metrics) ? metrics : []);
      } catch (e2) {
        setDetailError(e2.message);
      }
    } finally {
      setDetailLoading(false);
    }
  };

  const onDelete = async (report) => {
    if (!window.confirm(`确定要删除报告 #${report.id} 吗？此操作不可恢复。`)) return;
    setDeletingId(report.id);
    try {
      await deleteReport(report.id);
      setSelected((cur) => (cur && cur.id === report.id ? null : cur));
      // 刷新列表
      const data = await getReports({ patient_id: patientId.trim() });
      setReports(Array.isArray(data) ? data : (data && data.reports) || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const countMetrics = (r) => {
    if (Array.isArray(r.metrics)) return r.metrics.length;
    if (typeof r.metric_count === 'number') return r.metric_count;
    return '—';
  };

  return (
    <div>
      <div className="card">
        <h3 className="card-title">按患者查询报告</h3>
        <form onSubmit={search} className="form-inline">
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="患者编号，例如 P001"
          />
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? '查询中…' : '查询'}
          </button>
        </form>
        {error && <div className="alert alert-error">{error}</div>}
      </div>

      {reports && (
        <div className="card">
          <h3 className="card-title">报告列表（{reports.length} 条）</h3>
          {reports.length === 0 ? (
            <p className="muted">未找到该患者的报告。</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>类型</th>
                  <th>检查日期</th>
                  <th>科室</th>
                  <th>指标数</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr
                    key={r.id}
                    className={selected && selected.id === r.id ? 'row-selected' : ''}
                  >
                    <td>{r.id}</td>
                    <td>{r.report_type || '—'}</td>
                    <td>{r.exam_date || '—'}</td>
                    <td>{r.department || '—'}</td>
                    <td>{countMetrics(r)}</td>
                    <td className="cell-actions">
                      <button className="btn btn-small" onClick={() => openDetail(r)}>
                        {selected && selected.id === r.id ? '查看中…' : '查看详情'}
                      </button>
                      <button
                        className="btn btn-small btn-danger"
                        onClick={() => onDelete(r)}
                        disabled={deletingId === r.id}
                      >
                        {deletingId === r.id ? '删除中…' : '删除'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* 报告详情 */}
      {selected && (
        <div className="card">
          <h3 className="card-title">报告详情 #{selected.id}</h3>
          <div className="meta-line">
            <span>类型：{selected.report_type || '—'}</span>
            <span>检查日期：{selected.exam_date || '—'}</span>
            <span>科室：{selected.department || '—'}</span>
            <span>创建时间：{selected.created_at || '—'}</span>
          </div>

          {detailLoading && <p className="muted">正在加载指标…</p>}
          {detailError && <div className="alert alert-error">{detailError}</div>}

          {!detailLoading && !detailError && (
            detailMetrics.length > 0 ? (
              <table className="table">
                <thead>
                  <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>单位</th>
                    <th>参考范围</th>
                    <th>趋势</th>
                    <th>异常</th>
                    <th>页码</th>
                    <th>证据文本</th>
                    <th>source_id</th>
                    <th>bbox</th>
                    <th>bbox_normalized</th>
                  </tr>
                </thead>
                <tbody>
                  {detailMetrics.map((m, i) => (
                    <tr key={i}>
                      <td>{m.metric_name || '—'}</td>
                      <td>{m.metric_value ?? '—'}</td>
                      <td>{m.unit || '—'}</td>
                      <td>{m.reference_range || '—'}</td>
                      <td>{m.trend || '—'}</td>
                      <td>{abnormalBadge(m.abnormal_flag)}</td>
                      <td>{m.page_number ?? '—'}</td>
                      <td className="evidence-cell">{m.evidence_text || '—'}</td>
                      <td>{m.source_id ?? '—'}</td>
                      <td className="mono">{m.bbox ? JSON.stringify(m.bbox) : '—'}</td>
                      <td className="mono">{m.bbox_normalized ? JSON.stringify(m.bbox_normalized) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">该报告没有指标数据。</p>
            )
          )}
        </div>
      )}
    </div>
  );
}

// 异常标记徽章：H=偏高 L=偏低 N=正常
function abnormalBadge(flag) {
  if (!flag) return <span className="badge badge-normal">N</span>;
  const f = String(flag).toUpperCase();
  if (f === 'H' || f === 'HIGH' || f === '高') return <span className="badge badge-high">H</span>;
  if (f === 'L' || f === 'LOW' || f === '低') return <span className="badge badge-low">L</span>;
  if (f === 'N' || f === 'NORMAL' || f === '正常') return <span className="badge badge-normal">N</span>;
  return <span className="badge">{String(flag)}</span>;
}
