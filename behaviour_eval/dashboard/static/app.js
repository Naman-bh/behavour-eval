// Initialise Icons
lucide.createIcons();

// State
let currentProject = "all";
let trendChartInstance = null;
let modelChartInstance = null;

// DOM Elements
const views = document.querySelectorAll('.view');
const navLinks = document.querySelectorAll('.nav-links a');
const projectSelect = document.getElementById('project-select');
const modelFilter = document.getElementById('filter-model');
const outcomeFilter = document.getElementById('filter-outcome');

// Navigation
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('data-target');
        
        // Update active nav
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        
        // Update views
        views.forEach(view => view.classList.add('hidden'));
        document.getElementById(`view-${targetId}`).classList.remove('hidden');

        // Load data if needed
        if (targetId === 'overview') loadOverview();
        if (targetId === 'trajectories') loadTrajectories();
        if (targetId === 'compare') loadComparisons();
    });
});

projectSelect.addEventListener('change', (e) => {
    currentProject = e.target.value;
    refreshData();
});

modelFilter.addEventListener('change', loadTrajectories);
outcomeFilter.addEventListener('change', loadTrajectories);

// Initialization
async function init() {
    await fetchProjects();
    loadOverview();
}

async function refreshData() {
    const activeView = document.querySelector('.nav-links a.active').getAttribute('data-target');
    if (activeView === 'overview') loadOverview();
    if (activeView === 'trajectories') loadTrajectories();
    if (activeView === 'compare') loadComparisons();
}

// API Calls
async function fetchProjects() {
    try {
        const res = await fetch('/api/projects');
        const data = await res.json();
        
        const select = document.getElementById('project-select');
        select.innerHTML = '<option value="all">All Projects</option>';
        
        data.projects.forEach(p => {
            if (p !== 'default') {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p;
                select.appendChild(opt);
            }
        });
    } catch (e) {
        console.error("Failed to fetch projects", e);
    }
}

// Overview View
async function loadOverview() {
    try {
        const res = await fetch(`/api/stats?project=${currentProject}`);
        const data = await res.json();

        // Update stat cards
        document.getElementById('stat-total').textContent = data.total_trajectories;
        document.getElementById('stat-success-rate').textContent = `${(data.success_rate * 100).toFixed(1)}%`;
        document.getElementById('stat-latency').textContent = `${Math.round(data.avg_latency_ms || 0)}ms`;
        document.getElementById('stat-reasoning').textContent = `${data.with_reasoning_pct.toFixed(1)}%`;

        renderCharts(data);
        renderAvgScores(data.avg_scores);

    } catch (e) {
        console.error("Failed to load overview", e);
    }
}

function renderCharts(data) {
    const chartText = '#52617b';
    const chartGrid = '#e4eaf2';
    // Trend Chart
    const trendCtx = document.getElementById('trendChart').getContext('2d');
    const reversedTrends = [...data.trends].reverse();
    
    if (trendChartInstance) trendChartInstance.destroy();
    
    trendChartInstance = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: reversedTrends.map(t => t.day),
            datasets: [{
                label: 'Success Rate',
                data: reversedTrends.map(t => t.success_rate * 100),
                borderColor: '#2563eb',
                tension: 0.4,
                fill: true,
                backgroundColor: 'rgba(37, 99, 235, 0.10)',
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100, ticks: { color: chartText }, grid: { color: chartGrid } }, x: { ticks: { color: chartText }, grid: { display: false } } },
            plugins: { legend: { display: false } }
        }
    });

    // Model Chart
    const modelCtx = document.getElementById('modelChart').getContext('2d');
    if (modelChartInstance) modelChartInstance.destroy();

    modelChartInstance = new Chart(modelCtx, {
        type: 'bar',
        data: {
            labels: data.models.map(m => m.model_id),
            datasets: [{
                label: 'Success Rate (%)',
                data: data.models.map(m => m.success_rate * 100),
                backgroundColor: '#0f9f8c',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100, ticks: { color: chartText }, grid: { color: chartGrid } }, x: { ticks: { color: chartText }, grid: { display: false } } },
            plugins: { legend: { display: false } }
        }
    });
}

function renderAvgScores(scores) {
    const container = document.getElementById('avg-scores-container');
    container.innerHTML = '';

    for (const [metric, value] of Object.entries(scores)) {
        if (metric === 'latency') continue;
        
        const pct = (value * 100).toFixed(1);
        const el = document.createElement('div');
        el.className = 'score-item';
        el.innerHTML = `
            <div class="score-header">
                <span>${metric}</span>
                <span>${pct}%</span>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: ${pct}%"></div>
            </div>
        `;
        container.appendChild(el);
    }
}

// Trajectories View
async function loadTrajectories() {
    try {
        const params = new URLSearchParams({ project: currentProject });
        if (modelFilter.value) params.set('model', modelFilter.value);
        if (outcomeFilter.value) params.set('outcome', outcomeFilter.value);
        const res = await fetch(`/api/trajectories?${params}`);
        const data = await res.json();
        updateModelFilter(data.trajectories);
        
        const tbody = document.querySelector('#trajectories-table tbody');
        tbody.innerHTML = '';

        data.trajectories.forEach(t => {
            const tr = document.createElement('tr');
            tr.onclick = () => showTrajectoryModal(t.trajectory_id);
            
            const date = new Date(t.timestamp).toLocaleString();
            const tagClass = `tag ${t.outcome}`;
            
            tr.innerHTML = `
                <td>${date}</td>
                <td><span class="tag" style="background:#334155">${t.model_id}</span></td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${t.user_task}">${t.user_task}</td>
                <td><span class="${tagClass}">${t.outcome}</span></td>
                <td>${Math.round(t.latency_ms || 0)}ms</td>
                <td>${t.actions.length} action(s)</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load trajectories", e);
    }
}

function updateModelFilter(trajectories) {
    const selected = modelFilter.value;
    const models = [...new Set(trajectories.map(t => t.model_id))].sort();
    modelFilter.innerHTML = '<option value="">All Models</option>';
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        option.selected = model === selected;
        modelFilter.appendChild(option);
    });
}

async function showTrajectoryModal(id) {
    try {
        const res = await fetch(`/api/trajectory/${id}`);
        const t = await res.json();

        document.getElementById('modal-task').textContent = t.user_task;
        document.getElementById('modal-model').textContent = t.model_id;
        document.getElementById('modal-outcome').textContent = t.outcome;
        document.getElementById('modal-outcome').className = `tag ${t.outcome}`;
        document.getElementById('modal-latency').textContent = `${Math.round(t.latency_ms || 0)}ms`;

        // Reasoning
        document.getElementById('modal-reasoning').textContent = t.agent_reasoning || "No reasoning provided.";

        // Scores
        let scoresHtml = '';
        for (const [metric, val] of Object.entries(t.scores || {})) {
            if (metric === 'latency') continue;
            scoresHtml += `
                <div class="score-item">
                    <div class="score-header"><span>${metric}</span><span>${(val*100).toFixed(0)}%</span></div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${val*100}%"></div></div>
                </div>
            `;
        }
        document.getElementById('modal-scores').innerHTML = scoresHtml || "<p>No scores available.</p>";

        // Tools
        let toolsHtml = '';
        (t.tool_calls || []).forEach(tc => {
            toolsHtml += `
                <div class="tool-call">
                    <div class="tool-name">${tc.tool_name}</div>
                    <div style="font-size:0.875rem; color:#94a3b8;">Args: ${JSON.stringify(tc.arguments)}</div>
                    ${tc.error ? `<div style="color:#f87171; font-size:0.875rem">Error: ${tc.error}</div>` : ''}
                </div>
            `;
        });
        document.getElementById('modal-tools').innerHTML = toolsHtml || "<p>No tools called.</p>";

        // Actions
        let actionsHtml = '';
        (t.actions || []).forEach(a => {
            const dangerTag = a.is_destructive ? '<span class="tag failure">Destructive</span>' : '';
            actionsHtml += `
                <div class="action-item">
                    <strong>${a.action_type.toUpperCase()}</strong>: ${a.target} ${dangerTag}
                    <div style="font-size:0.875rem; color:#94a3b8; margin-top:0.25rem;">${a.details}</div>
                </div>
            `;
        });
        document.getElementById('modal-actions').innerHTML = actionsHtml || "<p>No actions taken.</p>";

        document.getElementById('trajectory-modal').classList.remove('hidden');
    } catch (e) {
        console.error("Failed to load trajectory details", e);
    }
}

function closeModal() {
    document.getElementById('trajectory-modal').classList.add('hidden');
}

// Compare Models View
async function loadComparisons() {
    try {
        const res = await fetch('/api/comparisons');
        const data = await res.json();
        
        const tbody = document.querySelector('#runs-table tbody');
        tbody.innerHTML = '';

        data.runs.forEach(run => {
            const tr = document.createElement('tr');
            tr.onclick = () => showComparisonDetail(run.run_id);
            
            const date = new Date(run.timestamp).toLocaleString();
            
            tr.innerHTML = `
                <td style="font-family:monospace">${run.run_id.split('_').pop()}</td>
                <td>${date}</td>
                <td>${run.scenario_set.split('/').pop()}</td>
                <td>${run.models.join(', ')}</td>
                <td><span class="tag ${run.status === 'completed' ? 'success' : 'partial'}">${run.status}</span></td>
                <td>${run.recommendation ? JSON.parse(run.recommendation).recommended || 'None' : '...'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load comparisons", e);
    }
}

async function showComparisonDetail(runId) {
    try {
        const res = await fetch(`/api/comparison/${runId}`);
        const data = await res.json();

        document.getElementById('comparison-detail').classList.remove('hidden');
        document.getElementById('run-id-display').textContent = runId;

        // Recommendation — stored as JSON string in DB, parse it
        let rec = data.recommendation;
        if (typeof rec === 'string') {
            try { rec = JSON.parse(rec); } catch (_) { rec = {}; }
        }
        rec = rec || {};
        let recHtml = `<h2>Recommendation: ${rec.recommended || 'Inconclusive'}</h2>`;
        recHtml += `<p>${rec.reasoning || ''}</p>`;
        document.getElementById('run-recommendation').innerHTML = recHtml;

        // Rankings — nested under data.results.rankings
        const rankings = data.results?.rankings?.rankings || [];
        const tbody = document.querySelector('#rankings-table tbody');
        tbody.innerHTML = '';
        rankings.forEach((r, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${idx + 1}</td>
                <td><strong>${r.model_id}</strong></td>
                <td>${r.elo_rating}</td>
                <td>${(r.win_rate * 100).toFixed(1)}%</td>
                <td>${r.wins} / ${r.losses} / ${r.ties}</td>
            `;
            tbody.appendChild(tr);
        });

        // Heatmap — models list from results or top-level
        const models = data.results?.models || data.models;
        renderHeatmap(data.win_rate_matrix, models);

    } catch (e) {
        console.error("Failed to load comparison detail", e);
    }
}

function renderHeatmap(matrix, models) {
    if (!matrix || !models) return;
    
    const container = document.getElementById('heatmap-container');
    container.style.gridTemplateColumns = `120px repeat(${models.length}, 1fr)`;
    container.innerHTML = '';

    // Header row
    container.appendChild(document.createElement('div')); // Empty corner
    models.forEach(m => {
        const el = document.createElement('div');
        el.style.textAlign = 'center';
        el.style.fontWeight = 'bold';
        el.textContent = m;
        container.appendChild(el);
    });

    // Rows
    models.forEach(m1 => {
        const rowLabel = document.createElement('div');
        rowLabel.style.fontWeight = 'bold';
        rowLabel.style.display = 'flex';
        rowLabel.style.alignItems = 'center';
        rowLabel.textContent = m1;
        container.appendChild(rowLabel);

        models.forEach(m2 => {
            const el = document.createElement('div');
            el.className = 'heatmap-cell';
            
            if (m1 === m2) {
                el.style.backgroundColor = 'var(--bg-dark)';
                el.textContent = '-';
            } else {
                const winRate = matrix[m1][m2];
                // Color scale: red (0%) to green (100%)
                const hue = Math.max(0, Math.min(120, winRate * 120));
                el.style.backgroundColor = `hsla(${hue}, 70%, 40%, 0.8)`;
                el.textContent = `${(winRate * 100).toFixed(0)}%`;
            }
            container.appendChild(el);
        });
    });
}

// Research View
async function loadResearch() {
    document.getElementById('rq1-loading').classList.remove('hidden');
    document.getElementById('rq2-loading').classList.remove('hidden');
    document.getElementById('rq1-content').classList.add('hidden');
    document.getElementById('rq2-content').classList.add('hidden');

    try {
        // Fetch RQ1
        const res1 = await fetch('/api/rq1/reasoning_analysis');
        const rq1 = await res1.json();
        
        document.getElementById('rq1-loading').classList.add('hidden');
        document.getElementById('rq1-content').classList.remove('hidden');
        
        document.getElementById('rq1-rule').innerHTML = `
            <strong>Weight Rule: ${rq1.principled_rule.weight}</strong><br>
            ${rq1.principled_rule.explanation}
        `;
        
        document.getElementById('rq1-avail').textContent = `${(rq1.availability.reasoning_availability_rate * 100).toFixed(1)}%`;
        document.getElementById('rq1-faith').textContent = `${(rq1.faithfulness.faithfulness_rate * 100).toFixed(1)}%`;
        document.getElementById('rq1-contra').textContent = `${(rq1.faithfulness.contradiction_rate * 100).toFixed(1)}%`;
        
        document.getElementById('rq1-summary').textContent = `Correlation with outcome: ${rq1.outcome_correlation.correlation_strength} (${(rq1.outcome_correlation.success_rate_delta * 100).toFixed(1)}% success delta)`;

        // Fetch RQ2
        const res2 = await fetch('/api/rq2/judge_reliability');
        const rq2 = await res2.json();
        
        document.getElementById('rq2-loading').classList.add('hidden');
        document.getElementById('rq2-content').classList.remove('hidden');

        document.getElementById('rq2-summary').textContent = rq2.summary;

        const tbody = document.getElementById('rq2-table');
        tbody.innerHTML = '';
        
        const metrics = ["task_completion", "tool_use_correctness", "intent_accuracy", "destructive_action_safety"];
        metrics.forEach(m => {
            const sj = rq2.single_judge_agreement[m]?.cohens_kappa || 0;
            const jj = rq2.jury_agreement[m]?.cohens_kappa || 0;
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${m}</td>
                <td>${sj.toFixed(3)}</td>
                <td><strong>${jj.toFixed(3)}</strong></td>
            `;
            tbody.appendChild(tr);
        });

        const hyp = document.getElementById('rq2-hypotheses');
        hyp.innerHTML = `
            <li><strong>H1 (Pairwise > Absolute):</strong> ${rq2.hypotheses.h1_pairwise_better_than_absolute ? '<span style="color:var(--secondary)">Supported</span>' : '<span style="color:var(--danger)">Not Supported</span>'} (Margin: ${(rq2.hypotheses.h1_margin * 100).toFixed(1)}%)</li>
            <li><strong>H2 (Jury > Single Judge):</strong> ${rq2.hypotheses.h2_jury_better_than_single ? '<span style="color:var(--secondary)">Supported</span>' : '<span style="color:var(--danger)">Not Supported</span>'} (Improvement: ${(rq2.hypotheses.h2_kappa_improvement).toFixed(3)} κ)</li>
        `;

    } catch (e) {
        console.error("Failed to load research", e);
    }
}

// Start
init();
