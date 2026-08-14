import React, { useMemo, useState } from 'react';
import { getMetricAnomalies, getMetricTrend, getMetricSearch } from '../api.js';

/* ---------------- 简易 SVG 折线图 ---------------- */
// 纯手工绘制的趋势图：x 轴为检查日期，y 轴为指标数值，异常点标红
function TrendChart({ dataPoints }) {
  const points = useMemo(() => {
    if (!Array.isArray(dataPoints)) return [];
    return dataPoints
      .map((p) => ({ ...p, num: Number(p.value) }))
      .filter((p) => Number.isFinite(p.num))
      .sort((a, b) => new Date(a.exam_date) - new Date(b.exam_date));
  }, [dataPoints]);

  if (points.length === 0) {
    return <p className="muted">暂无有效趋势数据（需要至少一个数值型数据点）。</p>;
  }

  const W = 680;
  const H = 280;
  const padL = 60;
  const padR = 30;
  const padT = 20;
  const padB = 42;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const values = points.map((p) => p.num);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  min -= span * 0.1;
  max += span * 0.1;

  const x = (i) => (points.length === 1 ? padL + innerW / 2 : padL + (i / (points.length - 1)) * innerW);
  const y = (v) => padT + innerH - ((v - min) / (max - min)) * innerH;

  const isAbnormal = (p) => {
    const f = String(p.abnormal_flag || '').toUpperCase();
    return f === 'H' || f === 'L' || f === 'HIGH' || f === 'LOW' || f === '异常' || f === '高' || f === '低';
  };

  // 横向网格线与 y 轴刻度
  const grid = [];
  const gridCount = 5;
  for (let i = 0; i <= gridCount; i += 1) {
    const gv = min + ((max - min) * i) / gridCount;
    grid.push({ y: y(gv), label: gv.toFixed(1) });
  }

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p.num).toFixed(1)}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="指标趋势折线图">
      {grid.map((g, i) => (
        <g key={i}>
          <line x1={padL} y1={g.y} x2={W - padR} y2={g.y} className="chart-grid" />
          <text x={padL - 8} y={g.y + 4} textAnchor="end" className="chart-label">{g.label}</text>
        </g>
      ))}
      <path d={linePath} fill="none" className="chart-line" />
      {points.map((p, i) => (
        <circle
          key={i}
          cx={x(i)}
          cy={y(p.num)}
          r={isAbnormal(p) ? 6 : 4.5}
          className={isAbnormal(p) ? 'point point-abnormal' : 'point point-normal'}
        >
          <title>
            {`${p.exam_date || ''} ${p.metric_name || ''}：${p.value} ${p.unit || ''}${isAbnormal(p) ? '（异常）' : ''}`}
          </title>
        </circle>
      ))}
      {points.map((p, i) => (
        <text key={`x${i}`} x={x(i)} y={H - padB + 18} textAnchor="middle" className="chart-label">
          {(p.exam_date || '').slice(0, 10)}
        </text>
      ))}
    </svg>
  );
}

// 异常标记徽章（H/L/N）
function abnormalBadge(flag) {
  if (!flag) return <span className="badge badge-normal">N</span>;
  const f = String(flag).toUpperCase();
  if (f === 'H' || f === 'HIGH' || f === '高') return <span className="badge badge-high">H</span>;
  if (f === 'L' || f === 'LOW' || f === '低') return <span className="badge badge-low">L</span>;
  if (f === 'N' || f === 'NORMAL' || f === '正常') return <span className="badge badge-normal">N</span>;
  return <span className="badge">{String(flag)}</span>;
}

export default function Metrics() {
  const [patientId, setPatientId] = useState('');
  const [days, setDays] = useState(90);

  // 异常指标
  const [anomalies, setAnomalies] = useState(null);
  const [anomalyError, setAnomalyError] = useState('');
  const [anomalyLoading, setAnomalyLoading] = useState(false);

  // 趋势：下拉选择 + 手动输入兜底
  const [metricName, setMetricName] = useState('');
  const [customMetric, setCustomMetric] = useState('');
  const [trend, setTrend] = useState(null);
  const [trendError, setTrendError] = useState('');
  const [trendLoading, setTrendLoading] = useState(false);

  // 指标搜索
  const [keyword, setKeyword] = useState('');
  const [abnormalOnly, setAbnormalOnly] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [searchError, setSearchError] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);

  // 从异常指标结果中提取不重复的指标名，作为趋势下拉的可选项
  const metricOptions = useMemo(() => {
    if (!anomalies || !Array.isArray(anomalies.anomalies)) return [];
    const names = anomalies.anomalies
      .map((a) => a.metric_name || a.metric || a.name)
      .filter(Boolean);
    return [...new Set(names)];
  }, [anomalies]);

  const effectiveMetric = metricName || customMetric.trim();

  const loadAnomalies = async () => {
    if (!patientId.trim()) { setAnomalyError('请先填写患者编号'); return; }
    setAnomalyError('');
    setAnomalyLoading(true);
    try {
      const data = await getMetricAnomalies({ patient_id: patientId.trim(), days });
      setAnomalies(data);
    } catch (err) {
      setAnomalyError(err.message);
      setAnomalies(null);
    } finally {
      setAnomalyLoading(false);
    }
  };

  const loadTrend = async () => {
    if (!patientId.trim()) { setTrendError('请先填写患者编号'); return; }
    if (!effectiveMetric) { setTrendError('请选择或输入指标名称'); return; }
    setTrendError('');
    setTrendLoading(true);
    try {
      const data = await getMetricTrend({ patient_id: patientId.trim(), metric_name: effectiveMetric, days });
      setTrend(data);
    } catch (err) {
      setTrendError(err.message);
      setTrend(null);
    } finally {
      setTrendLoading(false);
    }
  };

  const doSearch = async () => {
    if (!patientId.trim()) { setSearchError('请先填写患者编号'); return; }
    setSearchError('');
    setSearchLoading(true);
    try {
      const data = await getMetricSearch({
        patient_id: patientId.trim(),
        keyword: keyword.trim(),
        abnormal_only: abnormalOnly,
        limit: 50,
      });
      setSearchResults(data);
    } catch (err) {
      setSearchError(err.message);
      setSearchResults(null);
    } finally {
      setSearchLoading(false);
    }
  };

  const searchList = searchResults && Array.isArray(searchResults) ? searchResults : (searchResults && searchResults.results) || [];
  const anomalyList = (anomalies && anomalies.anomalies) || [];

  return (
    <div>
      {/* 患者编号设置（三个区块共用） */}
      <div className="card">
        <h3 className="card-title">患者与时间范围</h3>
        <div className="form-inline">
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            placeholder="患者编号，例如 P001"
          />
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            <option value={30}>近 30 天</option>
            <option value={90}>近 90 天</option>
            <option value={180}>近 180 天</option>
            <option value={365}>近 365 天</option>
          </select>
        </div>
      </div>

      {/* 一、异常指标汇总 */}
      <div className="card">
        <div className="card-title-row">
          <h3 className="card-title">异常指标汇总</h3>
          <button className="btn btn-small" onClick={loadAnomalies} disabled={anomalyLoading}>
            {anomalyLoading ? '加载中…' : '加载'}
          </button>
        </div>
        {anomalyError && <div className="alert alert-error">{anomalyError}</div>}
        {anomalies && (
          <>
            {anomalies.summary && <p className="hint">摘要：{anomalies.summary}</p>}
            {anomalyList.length > 0 ? (
              <table className="table">
                <thead>
                  <tr>
                    <th>指标名称</th>
                    <th>数值</th>
                    <th>单位</th>
                    <th>参考范围</th>
                    <th>异常</th>
                    <th>检查日期</th>
                    <th>报告 ID</th>
                  </tr>
                </thead>
                <tbody>
                  {anomalyList.map((a, i) => (
                    <tr key={i}>
                      <td>{a.metric_name || a.metric || a.name || '—'}</td>
                      <td>{a.metric_value ?? a.value ?? '—'}</td>
                      <td>{a.unit || '—'}</td>
                      <td>{a.reference_range || '—'}</td>
                      <td>{abnormalBadge(a.abnormal_flag)}</td>
                      <td>{a.exam_date || a.date || '—'}</td>
                      <td>{a.report_id ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">未发现异常指标。</p>
            )}
          </>
        )}
      </div>

      {/* 二、指标趋势 */}
      <div className="card">
        <div className="card-title-row">
          <h3 className="card-title">指标趋势</h3>
          <button className="btn btn-small" onClick={loadTrend} disabled={trendLoading}>
            {trendLoading ? '加载中…' : '查询趋势'}
          </button>
        </div>
        <div className="form-inline">
          <select value={metricName} onChange={(e) => setMetricName(e.target.value)}>
            <option value="">选择指标…</option>
            {metricOptions.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <input
            type="text"
            value={customMetric}
            onChange={(e) => setCustomMetric(e.target.value)}
            placeholder="或手动输入指标名称"
          />
        </div>
        {trendError && <div className="alert alert-error">{trendError}</div>}
        {trend && (
          <>
            <p className="hint">
              指标：{trend.metric_name} ｜ 数据点：{trend.data_points ? trend.data_points.length : 0} 个
              {trend.statistics && (
                <> ｜ 均值 {trend.statistics.average ?? '—'} ｜ 最高 {trend.statistics.max ?? '—'} ｜ 最低 {trend.statistics.min ?? '—'}</>
              )}
            </p>
            <TrendChart dataPoints={trend.data_points} />
            {trend.data_points && trend.data_points.length > 0 && (
              <table className="table">
                <thead>
                  <tr>
                    <th>检查日期</th>
                    <th>数值</th>
                    <th>单位</th>
                    <th>参考范围</th>
                    <th>趋势</th>
                    <th>异常</th>
                    <th>报告 ID</th>
                  </tr>
                </thead>
                <tbody>
                  {trend.data_points.map((p, i) => (
                    <tr key={i}>
                      <td>{p.exam_date || '—'}</td>
                      <td>{p.value ?? '—'}</td>
                      <td>{p.unit || '—'}</td>
                      <td>{p.reference_range || '—'}</td>
                      <td>{p.trend || '—'}</td>
                      <td>{abnormalBadge(p.abnormal_flag)}</td>
                      <td>{p.report_id ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </div>

      {/* 三、指标搜索 */}
      <div className="card">
        <div className="card-title-row">
          <h3 className="card-title">指标搜索</h3>
          <button className="btn btn-small" onClick={doSearch} disabled={searchLoading}>
            {searchLoading ? '搜索中…' : '搜索'}
          </button>
        </div>
        <div className="form-inline">
          <input
            type="text"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="输入关键词，例如 白细胞"
            onKeyDown={(e) => { if (e.key === 'Enter') doSearch(); }}
          />
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={abnormalOnly}
              onChange={(e) => setAbnormalOnly(e.target.checked)}
            />
            仅异常
          </label>
        </div>
        {searchError && <div className="alert alert-error">{searchError}</div>}
        {searchResults && (
          searchList.length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>指标名称</th>
                  <th>数值</th>
                  <th>单位</th>
                  <th>参考范围</th>
                  <th>异常</th>
                  <th>检查日期</th>
                  <th>报告 ID</th>
                </tr>
              </thead>
              <tbody>
                {searchList.map((m, i) => (
                  <tr key={i}>
                    <td>{m.metric_name || m.metric || '—'}</td>
                    <td>{m.metric_value ?? m.value ?? '—'}</td>
                    <td>{m.unit || '—'}</td>
                    <td>{m.reference_range || '—'}</td>
                    <td>{abnormalBadge(m.abnormal_flag)}</td>
                    <td>{m.exam_date || '—'}</td>
                    <td>{m.report_id ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">没有匹配的指标。</p>
          )
        )}
      </div>
    </div>
  );
}
