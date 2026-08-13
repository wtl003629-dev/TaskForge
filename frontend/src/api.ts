import type {
  AgentProfile,
  ApiResult,
  AuditEvent,
  AuditUsage,
  CreateReviewCaseInput,
  CreateRunInput,
  Evidence,
  IngestionStatus,
  LiteratureDiscoveryResult,
  LiteratureProviderReport,
  LiteratureSearchInput,
  McpServerSummary,
  MetricsSnapshot,
  OperationJob,
  PendingApproval,
  PaperCard,
  ReviewCase,
  ReviewCaseAuditEvent,
  ReviewCaseDetail,
  ReviewCaseList,
  ReviewDecisionInput,
  ReviewEvidenceRef,
  ReviewExecutionDisclosure,
  ReviewFact,
  ReviewHandoff,
  ReviewHumanDecision,
  ReviewPlan,
  ReviewPlanSlot,
  ReviewRecommendation,
  ReviewRoleRun,
  ReviewSubmission,
  ResearchEvidenceCard,
  ResearchScope,
  RetrievalConfidence,
  RunRecord,
  RunStep,
  SkillPack,
  ScopeEvidenceResult,
  ToolCall,
} from './types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')
const MOCK_ENABLED = import.meta.env.VITE_ENABLE_MOCK_FALLBACK !== 'false'

class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message)
  }
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    let message = detail
    try {
      const parsed = JSON.parse(detail) as { detail?: unknown }
      if (typeof parsed.detail === 'string') {
        message = parsed.detail
      } else if (Array.isArray(parsed.detail)) {
        message = parsed.detail
          .map((item) => {
            const error = asRecord(item)
            const location = asArray(error.loc).map((part) => String(part)).join('.')
            return `${location ? `${location}: ` : ''}${text(error.msg, '请求参数无效')}`
          })
          .join('；')
      }
    } catch {
      // Non-JSON error bodies remain useful as-is.
    }
    throw new ApiError(message || `API 请求失败：${response.status}`, response.status)
  }

  if (response.status === 204) return null
  return response.json()
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function text(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function bool(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringArray(value: unknown): string[] {
  return asArray(value).map((item) => text(item)).filter(Boolean)
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function normalizeSkillPack(value: unknown, index: number): SkillPack {
  const item = asRecord(value)
  const tools = asArray(item.tools).map((tool) => text(tool)).filter(Boolean)
  return {
    id: text(item.id ?? item.slug, `skill-${index + 1}`),
    name: text(item.name ?? item.label, `Skill Pack ${index + 1}`),
    description: text(item.description),
    tools,
  }
}

function normalizeAgent(value: unknown, index: number): AgentProfile {
  const item = asRecord(value)
  const metadata = asRecord(item.metadata)
  const packs = item.skill_packs ?? item.skillPacks ?? item.skills ?? metadata.skill_packs
  const normalizedPacks = asArray(packs).map(normalizeSkillPack)
  const allowedTools = asArray(item.allowed_tools ?? item.allowedTools)
    .map((tool) => text(tool))
    .filter(Boolean)
  return {
    id: text(item.id ?? item.slug, `agent-${index + 1}`),
    name: text(item.name ?? item.label, `Agent ${index + 1}`),
    description: text(
      item.description ?? metadata.description,
      '该 Profile 的指令与能力边界由后端配置。',
    ),
    skillPacks:
      normalizedPacks.length || !allowedTools.length
        ? normalizedPacks
        : [
            {
              id: 'default',
              name: '默认能力包',
              description: '由 Agent Profile 的 allowed_tools 生成。',
              tools: allowedTools,
            },
          ],
  }
}

function normalizeEvidence(value: unknown, index: number): Evidence {
  const item = asRecord(value)
  return {
    id: text(item.id, `evidence-${index + 1}`),
    title: text(item.title ?? item.name, `证据 ${index + 1}`),
    kind: text(item.kind ?? item.type, 'artifact'),
    source: text(item.source ?? item.uri) || undefined,
    summary: text(item.summary ?? item.content ?? item.value),
  }
}

function normalizeToolCall(value: unknown, index: number): ToolCall {
  const item = asRecord(value)
  return {
    id: text(item.id ?? item.call_id, `tool-${index + 1}`),
    name: text(item.name ?? item.tool_name, 'unknown_tool'),
    status: text(item.status, 'unknown'),
    arguments: objectValue(item.arguments ?? item.args),
    result: text(item.result ?? item.output) || undefined,
    requiresApproval: bool(item.requires_approval ?? item.requiresApproval),
  }
}

function normalizeStepToolCalls(item: Record<string, unknown>): ToolCall[] {
  const direct = asArray(item.tool_calls ?? item.toolCalls)
  if (direct.length) return direct.map(normalizeToolCall)

  const modelTurn = asRecord(item.model_turn ?? item.modelTurn)
  const requests = asArray(modelTurn.tool_requests ?? modelTurn.toolRequests)
  const results = asArray(item.tool_results ?? item.toolResults).map(asRecord)
  return requests.map((requestValue, index) => {
    const request = asRecord(requestValue)
    const callId = text(request.call_id ?? request.id, `tool-${index + 1}`)
    const result = results.find((candidate) => text(candidate.call_id) === callId)
    const ok = result ? bool(result.ok) : undefined
    const rawResult = result?.output ?? result?.error
    return {
      id: callId,
      name: text(request.name, 'unknown_tool'),
      status: result ? (ok ? 'completed' : 'failed') : text(item.status, 'proposed'),
      arguments: objectValue(request.arguments ?? request.args),
      result:
        rawResult === undefined
          ? undefined
          : typeof rawResult === 'string'
            ? rawResult
            : JSON.stringify(rawResult),
      requiresApproval: text(item.status) === 'waiting_approval',
    }
  })
}

function normalizeStep(value: unknown, index: number): RunStep {
  const item = asRecord(value)
  const stepIndex = typeof item.index === 'number' ? item.index + 1 : index + 1
  return {
    id: text(item.id, `step-${index + 1}`),
    title: text(item.title ?? item.name, `执行步骤 ${stepIndex}`),
    status: text(item.status, 'unknown'),
    summary: text(item.summary ?? item.safe_summary ?? item.description) || undefined,
    toolCalls: normalizeStepToolCalls(item),
    evidence: asArray(item.evidence).map(normalizeEvidence),
  }
}

function normalizeApproval(value: unknown): PendingApproval | undefined {
  if (!value) return undefined
  const item = asRecord(value)
  const approvalRequest = asRecord(item.request)
  return {
    id: text(item.id ?? item.approval_id ?? approvalRequest.call_id, 'pending-approval'),
    toolName: text(
      item.tool_name ?? item.toolName ?? item.name ?? approvalRequest.name,
      'sensitive_tool',
    ),
    reason: text(item.reason, '该操作需要人工确认。'),
    risk: text(item.risk ?? item.risk_level, '需检查副作用'),
    arguments: objectValue(item.arguments ?? item.args ?? approvalRequest.arguments),
  }
}

function normalizeRun(value: unknown): RunRecord {
  const root = asRecord(value)
  const item = asRecord(root.run ?? value)
  const now = new Date().toISOString()
  return {
    id: text(item.id ?? item.run_id, 'unknown-run'),
    task: text(item.task ?? item.goal ?? item.task_id),
    agentId: text(item.agent_id ?? item.agentId ?? item.agent_profile_id),
    skillPackId: text(item.skill_pack_id ?? item.skillPackId ?? item.skill_pack),
    status: text(item.status, 'queued'),
    createdAt: text(item.created_at ?? item.createdAt, now),
    updatedAt: text(item.updated_at ?? item.updatedAt, now),
    steps: asArray(item.steps).map(normalizeStep),
    evidence: asArray(item.evidence ?? item.artifacts).map(normalizeEvidence),
    pendingApproval: normalizeApproval(item.pending_approval ?? item.pendingApproval),
    summary: text(item.summary ?? item.final_answer) || undefined,
  }
}

function normalizeJob(value: unknown): OperationJob {
  const item = asRecord(value)
  const now = new Date().toISOString()
  return {
    runId: text(item.run_id ?? item.runId, 'unknown-run'),
    status: text(item.status, 'queued'),
    attempt: numberValue(item.attempt) ?? 0,
    maxAttempts: numberValue(item.max_attempts ?? item.maxAttempts) ?? 1,
    availableAt: text(item.available_at ?? item.availableAt, now),
    leaseExpiresAt: text(item.lease_expires_at ?? item.leaseExpiresAt) || undefined,
    resultStatus: text(item.result_status ?? item.resultStatus) || undefined,
    lastError: text(item.last_error ?? item.lastError) || undefined,
    updatedAt: text(item.updated_at ?? item.updatedAt, now),
  }
}

function normalizeUsage(value: unknown): AuditUsage | undefined {
  if (!value) return undefined
  const item = asRecord(value)
  return {
    inputTokens: numberValue(item.input_tokens ?? item.inputTokens),
    outputTokens: numberValue(item.output_tokens ?? item.outputTokens),
    totalTokens: numberValue(item.total_tokens ?? item.totalTokens),
    costUsd: numberValue(item.cost_usd ?? item.costUsd),
  }
}

function normalizeAuditEvent(value: unknown, index: number): AuditEvent {
  const item = asRecord(value)
  return {
    eventId: text(item.event_id ?? item.eventId, `audit-${index + 1}`),
    action: text(item.action, 'unknown'),
    outcome: text(item.outcome, 'unknown'),
    occurredAt: text(item.occurred_at ?? item.occurredAt, new Date().toISOString()),
    durationMs: numberValue(item.duration_ms ?? item.durationMs),
    tool: text(item.tool) || undefined,
    provider: text(item.provider) || undefined,
    usage: normalizeUsage(item.usage),
    safetyViolation: bool(item.safety_violation ?? item.safetyViolation) ?? false,
  }
}

function normalizeMetrics(value: unknown): MetricsSnapshot {
  const item = asRecord(value)
  return {
    runId: text(item.run_id ?? item.runId) || undefined,
    runCount: numberValue(item.run_count ?? item.runCount) ?? 0,
    runSuccessCount: numberValue(item.run_success_count ?? item.runSuccessCount) ?? 0,
    runSuccessRate: numberValue(item.run_success_rate ?? item.runSuccessRate),
    toolCount: numberValue(item.tool_count ?? item.toolCount) ?? 0,
    toolSuccessCount: numberValue(item.tool_success_count ?? item.toolSuccessCount) ?? 0,
    toolSuccessRate: numberValue(item.tool_success_rate ?? item.toolSuccessRate),
    durationP50Ms: numberValue(item.duration_p50_ms ?? item.durationP50Ms),
    durationP95Ms: numberValue(item.duration_p95_ms ?? item.durationP95Ms),
    inputTokens: numberValue(item.input_tokens ?? item.inputTokens),
    outputTokens: numberValue(item.output_tokens ?? item.outputTokens),
    totalTokens: numberValue(item.total_tokens ?? item.totalTokens),
    costUsd: numberValue(item.cost_usd ?? item.costUsd),
    safetyViolationCount:
      numberValue(item.safety_violation_count ?? item.safetyViolationCount) ?? 0,
  }
}

function normalizeMcpServer(value: unknown): McpServerSummary {
  const item = asRecord(value)
  return {
    namespace: text(item.namespace, 'unnamed'),
    enabled: bool(item.enabled) ?? false,
    profileIds: stringArray(item.profile_ids ?? item.profileIds),
    configuredTools: stringArray(item.configured_tools ?? item.configuredTools),
    mountedTools: stringArray(item.mounted_tools ?? item.mountedTools),
  }
}

function normalizeReviewEvidence(value: unknown): ReviewEvidenceRef {
  const item = asRecord(value)
  return {
    evidenceId: text(item.evidence_id, 'unknown-evidence'),
    sourceType: text(item.source_type, 'document'),
    locator: text(item.locator),
    excerpt: text(item.excerpt),
    title: text(item.title) || undefined,
    version: text(item.version) || undefined,
    checksumSha256: text(item.checksum_sha256) || undefined,
    pageNumber: numberValue(item.page_number),
  }
}

function normalizeReviewSubmission(value: unknown): ReviewSubmission {
  const item = asRecord(value)
  return {
    requestSummary: text(item.request_summary),
    businessJustification: text(item.business_justification),
    attributes: objectValue(item.attributes) ?? {},
    evidenceRefs: asArray(item.evidence_refs).map(normalizeReviewEvidence),
  }
}

function normalizeReviewRecommendation(value: unknown): ReviewRecommendation | undefined {
  if (!value) return undefined
  const item = asRecord(value)
  return {
    recommendationId: text(item.recommendation_id),
    modelRunId: text(item.model_run_id),
    modelId: text(item.model_id),
    outcome: text(item.outcome, 'escalate'),
    summary: text(item.summary),
    rationale: text(item.rationale),
    confidence: numberValue(item.confidence) ?? 0,
    evidenceRefs: asArray(item.evidence_refs).map(normalizeReviewEvidence),
    authority: 'model_untrusted',
    producedAt: text(item.produced_at),
  }
}

function normalizeReviewHumanDecision(value: unknown): ReviewHumanDecision | undefined {
  if (!value) return undefined
  const item = asRecord(value)
  const actor = asRecord(item.actor)
  const outcome = text(item.outcome)
  if (outcome !== 'approved' && outcome !== 'rejected') return undefined
  return {
    outcome,
    actor: {
      displayName: text(actor.display_name) || undefined,
      authority: 'human',
    },
    rationale: text(item.rationale),
    evidenceRefIds: stringArray(item.evidence_ref_ids),
    decidedAt: text(item.decided_at),
  }
}

function normalizeReviewCase(value: unknown): ReviewCase {
  const item = asRecord(value)
  const failure = asRecord(item.failure)
  return {
    caseId: text(item.case_id, 'unknown-case'),
    conversationId: text(item.conversation_id),
    kind: text(item.kind, 'enterprise_change') as ReviewCase['kind'],
    title: text(item.title, '未命名审查'),
    submission: normalizeReviewSubmission(item.submission),
    status: text(item.status, 'draft'),
    recommendation: normalizeReviewRecommendation(item.recommendation),
    humanDecision: normalizeReviewHumanDecision(item.human_decision),
    failure: item.failure
      ? { reason: text(failure.reason), failedAt: text(failure.failed_at) }
      : undefined,
    revision: numberValue(item.revision) ?? 1,
    createdAt: text(item.created_at),
    updatedAt: text(item.updated_at),
    submittedAt: text(item.submitted_at) || undefined,
    startedAt: text(item.started_at) || undefined,
    reviewRequestedAt: text(item.review_requested_at) || undefined,
    resolvedAt: text(item.resolved_at) || undefined,
  }
}

function normalizeReviewDisclosure(value: unknown): ReviewExecutionDisclosure {
  const item = asRecord(value)
  const mode = text(item.mode, 'injected-test-provider')
  return {
    provider: text(item.provider, 'unknown'),
    mode: (
      ['offline-deterministic-demo', 'configured-provider', 'injected-test-provider'].includes(mode)
        ? mode
        : 'injected-test-provider'
    ) as ReviewExecutionDisclosure['mode'],
    providerConfigured: bool(item.provider_configured) ?? false,
    contractTestedMock: bool(item.contract_tested_mock) ?? false,
    liveSmokeVerified: bool(item.live_smoke_verified) ?? false,
    businessE2eVerified: bool(item.business_e2e_verified) ?? false,
    recommendationAuthority: 'model_untrusted',
    finalDecisionAuthority: 'human',
  }
}

function normalizeReviewSlot(value: unknown): ReviewPlanSlot {
  const item = asRecord(value)
  return {
    slotId: text(item.slot_id),
    roleId: text(item.role_id),
    agentProfileId: text(item.agent_profile_id),
    dependsOn: stringArray(item.depends_on),
    order: numberValue(item.order) ?? 0,
    required: bool(item.required) ?? true,
    maxAttempts: numberValue(item.max_attempts) ?? 1,
  }
}

function normalizeReviewPlan(value: unknown): ReviewPlan | undefined {
  if (!value) return undefined
  const item = asRecord(value)
  return {
    planId: text(item.plan_id),
    status: text(item.status),
    version: numberValue(item.version) ?? 1,
    slots: asArray(item.slots).map(normalizeReviewSlot),
    createdAt: text(item.created_at),
    updatedAt: text(item.updated_at),
  }
}

function normalizeReviewRoleRun(value: unknown): ReviewRoleRun {
  const item = asRecord(value)
  const metrics = objectValue(item.runtime_metrics)
  const usage = objectValue(metrics?.usage)
  return {
    roleRunId: text(item.role_run_id),
    slotId: text(item.slot_id),
    roleId: text(item.role_id),
    agentProfileId: text(item.agent_profile_id),
    attempt: numberValue(item.attempt) ?? 1,
    status: text(item.status, 'pending'),
    version: numberValue(item.version) ?? 1,
    runtimeStatus: text(item.runtime_status) || undefined,
    summary: text(item.summary) || undefined,
    summaryAuthority: text(item.summary_authority) || undefined,
    citations: stringArray(item.citations),
    retrievedEvidenceRefs: stringArray(item.retrieved_evidence_refs),
    runtimeMetrics: metrics
      ? {
          stepCount: numberValue(metrics.step_count) ?? 0,
          modelTurnCount: numberValue(metrics.model_turn_count) ?? 0,
          toolCallCount: numberValue(metrics.tool_call_count) ?? 0,
          toolResultCount: numberValue(metrics.tool_result_count) ?? 0,
          toolSuccessCount: numberValue(metrics.tool_success_count) ?? 0,
          toolFailureCount: numberValue(metrics.tool_failure_count) ?? 0,
          safetyViolationCount: numberValue(metrics.safety_violation_count) ?? 0,
          elapsedMs: numberValue(metrics.elapsed_ms) ?? 0,
          usage: usage
            ? {
                inputTokens: numberValue(usage.input_tokens),
                outputTokens: numberValue(usage.output_tokens),
                totalTokens: numberValue(usage.total_tokens),
                costUsd: numberValue(usage.cost_usd),
              }
            : undefined,
        }
      : undefined,
    pendingApprovalCallId: text(item.pending_approval_call_id) || undefined,
    roleResult: objectValue(item.role_result),
    error: text(item.error) || undefined,
    createdAt: text(item.created_at),
    updatedAt: text(item.updated_at),
  }
}

function normalizeReviewFact(value: unknown): ReviewFact {
  const item = asRecord(value)
  return {
    factId: text(item.fact_id),
    factKey: text(item.fact_key),
    value: item.value,
    status: text(item.status, 'proposed'),
    authority: text(item.authority, 'model'),
    version: numberValue(item.version) ?? 1,
    sourceRoleRunId: text(item.source_role_run_id) || undefined,
    verifierRef: text(item.verifier_ref) || undefined,
    createdAt: text(item.created_at),
  }
}

function normalizeReviewHandoff(value: unknown): ReviewHandoff {
  const item = asRecord(value)
  return {
    handoffId: text(item.handoff_id, 'unknown-handoff'),
    fromRoleRunId: text(item.from_role_run_id),
    toSlotId: text(item.to_slot_id),
    summary: text(item.summary),
    sharedFactIds: stringArray(item.shared_fact_ids),
    createdAt: text(item.created_at),
  }
}

function normalizeReviewAudit(value: unknown): ReviewCaseAuditEvent {
  const item = asRecord(value)
  return {
    eventId: text(item.event_id),
    eventType: text(item.event_type),
    revision: numberValue(item.revision) ?? 1,
    fromStatus: text(item.from_status) || undefined,
    toStatus: text(item.to_status),
    actorAuthority: text(item.actor_authority),
    details: objectValue(item.details) ?? {},
    createdAt: text(item.created_at),
  }
}

function normalizeReviewDetail(value: unknown): ReviewCaseDetail {
  const item = asRecord(value)
  return {
    case: normalizeReviewCase(item.case),
    plan: normalizeReviewPlan(item.plan),
    roleRuns: asArray(item.role_runs).map(normalizeReviewRoleRun),
    sharedFacts: asArray(item.shared_facts).map(normalizeReviewFact),
    handoffs: asArray(item.handoffs).map(normalizeReviewHandoff),
    auditEvents: asArray(item.audit_events).map(normalizeReviewAudit),
    execution: normalizeReviewDisclosure(item.execution),
  }
}

const demoAgents: AgentProfile[] = [
  {
    id: 'general-agent',
    name: '通用任务执行器',
    description: '演示 Profile：由宿主治理工具、审批和执行轨迹，不代表真实 Provider 已接通。',
    skillPacks: [
      {
        id: 'research',
        name: '研究与报告',
        description: '检索证据、汇总来源并生成报告草稿。',
        tools: ['search_knowledge', 'read_document', 'write_artifact'],
      },
      {
        id: 'workspace',
        name: '工作区分析',
        description: '只读检索工作区并输出诊断证据。',
        tools: ['list_files', 'search_code', 'read_file'],
      },
    ],
  },
  {
    id: 'document-agent',
    name: '文档处理 Agent',
    description: '演示 Profile：针对文档抽取、比较和结构化交付。',
    skillPacks: [
      {
        id: 'document-review',
        name: '文档审阅',
        description: '抽取关键段落、比较差异并保留来源。',
        tools: ['read_document', 'extract_sections', 'write_artifact'],
      },
    ],
  },
]

interface MockState {
  run: RunRecord
  ticks: number
  executionMode: CreateRunInput['executionMode']
}

const mockRuns = new Map<string, MockState>()

function mockWarning(error: unknown): string {
  const reason = error instanceof Error ? error.message : '未知网络错误'
  return `后端 API 不可用（${reason}）。当前展示本地演示数据，不代表任务已真实执行。`
}

function mockCreateRun(input: CreateRunInput): RunRecord {
  const now = new Date().toISOString()
  const id = `demo-${Date.now().toString(36)}`
  const run: RunRecord = {
    id,
    task: input.task,
    agentId: input.agentId,
    skillPackId: input.skillPackId,
    status: 'queued',
    createdAt: now,
    updatedAt: now,
    steps: [
      {
        id: 'step-plan',
        title: '建立执行计划',
        status: 'queued',
        summary: '等待本地演示运行器推进。',
        toolCalls: [],
        evidence: [],
      },
    ],
    evidence: [],
  }
  mockRuns.set(id, { run, ticks: 0, executionMode: input.executionMode })
  return run
}

function advanceMockRun(id: string): RunRecord {
  const state = mockRuns.get(id)
  if (!state) throw new ApiError(`未找到演示 Run：${id}`, 404)
  if (state.run.status === 'waiting_approval' || ['completed', 'failed'].includes(state.run.status)) {
    return structuredClone(state.run)
  }

  state.ticks += 1
  state.run.updatedAt = new Date().toISOString()
  if (state.ticks === 1) {
    state.run.status = 'running'
    state.run.steps = [
      {
        id: 'step-plan',
        title: '建立执行计划',
        status: 'completed',
        summary: '已将目标拆成检索、核验和交付三个阶段。',
        toolCalls: [],
        evidence: [],
      },
      {
        id: 'step-search',
        title: '检索相关证据',
        status: 'running',
        summary: '正在调用只读检索工具。',
        toolCalls: [
          {
            id: 'call-search',
            name: 'search_knowledge',
            status: 'completed',
            arguments: { query: state.run.task, limit: 5 },
            result: '演示结果：找到 3 条候选证据。',
          },
        ],
        evidence: [],
      },
    ]
  } else {
    state.run.status = 'waiting_approval'
    state.run.steps[1].status = 'completed'
    state.run.steps.push({
      id: 'step-artifact',
      title: '生成交付物',
      status: 'waiting_approval',
      summary: '写入 Artifact 属于有副作用操作，等待人工确认。',
      toolCalls: [
        {
          id: 'call-write',
          name: 'write_artifact',
          status: 'waiting_approval',
          arguments: { path: 'artifacts/demo-report.md' },
          requiresApproval: true,
        },
      ],
      evidence: [],
    })
    state.run.pendingApproval = {
      id: 'approval-write',
      toolName: 'write_artifact',
      reason: '该操作将创建演示 Artifact。请核对目标路径和任务范围。',
      risk: '可逆的工作区写入（本地 mock 不会实际写文件）',
      arguments: { path: 'artifacts/demo-report.md' },
    }
  }
  return structuredClone(state.run)
}

function mockApprove(id: string, approved: boolean): RunRecord {
  const state = mockRuns.get(id)
  if (!state) throw new ApiError(`未找到演示 Run：${id}`, 404)
  const step = state.run.steps.find((item) => item.id === 'step-artifact')
  const call = step?.toolCalls.find((item) => item.id === 'call-write')
  state.run.pendingApproval = undefined
  state.run.updatedAt = new Date().toISOString()

  if (!approved) {
    state.run.status = 'failed'
    state.run.summary = '人工拒绝了写入操作，演示 Run 已停止。'
    if (step) step.status = 'failed'
    if (call) call.status = 'rejected'
    return structuredClone(state.run)
  }

  const evidence: Evidence = {
    id: 'evidence-demo',
    title: '演示研究摘要',
    kind: 'artifact',
    source: 'mock://artifacts/demo-report.md',
    summary: '本条证据由前端 mock 生成，仅用于演示 Evidence UI。',
  }
  state.run.status = 'completed'
  state.run.summary = '演示流程已完成；未调用真实模型、工具或持久化服务。'
  state.run.evidence = [evidence]
  if (step) {
    step.status = 'completed'
    step.summary = '审批通过，已生成模拟 Artifact。'
    step.evidence = [evidence]
  }
  if (call) {
    call.status = 'completed'
    call.result = 'mock://artifacts/demo-report.md'
  }
  return structuredClone(state.run)
}

function mockJob(id: string): OperationJob {
  const state = mockRuns.get(id)
  if (!state || state.executionMode !== 'queued') {
    throw new ApiError(`未找到演示队列任务：${id}`, 404)
  }
  const terminal = ['waiting_approval', 'completed', 'failed'].includes(state.run.status)
  return {
    runId: id,
    status: terminal ? 'completed' : state.ticks > 0 ? 'leased' : 'queued',
    attempt: state.ticks > 0 ? 1 : 0,
    maxAttempts: 3,
    availableAt: state.run.createdAt,
    leaseExpiresAt: state.ticks > 0 && !terminal
      ? new Date(Date.now() + 30_000).toISOString()
      : undefined,
    resultStatus: terminal ? state.run.status : undefined,
    updatedAt: state.run.updatedAt,
  }
}

function mockAudit(id: string): AuditEvent[] {
  const state = mockRuns.get(id)
  if (!state) throw new ApiError(`未找到演示 Run：${id}`, 404)
  const events: AuditEvent[] = []
  if (state.executionMode === 'queued') {
    events.push({
      eventId: `${id}-enqueue`,
      action: 'run.enqueue',
      outcome: 'queued',
      occurredAt: state.run.createdAt,
      safetyViolation: false,
    })
  }
  if (state.ticks > 0) {
    events.push({
      eventId: `${id}-search`,
      action: 'tool.execute',
      outcome: 'completed',
      occurredAt: state.run.updatedAt,
      durationMs: 42,
      tool: 'search_knowledge',
      safetyViolation: false,
    })
  }
  if (['waiting_approval', 'completed', 'failed'].includes(state.run.status)) {
    events.push({
      eventId: `${id}-run-state`,
      action: 'run.demo',
      outcome: state.run.status,
      occurredAt: state.run.updatedAt,
      durationMs: 120,
      provider: 'frontend-mock',
      safetyViolation: false,
    })
  }
  return events
}

function mockMetrics(id: string): MetricsSnapshot {
  const events = mockAudit(id)
  const toolCount = events.filter((event) => event.tool).length
  const runFinished = events.some((event) => event.action === 'run.demo')
  const state = mockRuns.get(id)
  const runSucceeded = state?.run.status === 'completed'
  return {
    runId: id,
    runCount: runFinished ? 1 : 0,
    runSuccessCount: runSucceeded ? 1 : 0,
    runSuccessRate: runFinished ? (runSucceeded ? 1 : 0) : undefined,
    toolCount,
    toolSuccessCount: toolCount,
    toolSuccessRate: toolCount ? 1 : undefined,
    durationP50Ms: runFinished ? 120 : undefined,
    durationP95Ms: runFinished ? 120 : undefined,
    safetyViolationCount: 0,
  }
}

function fallback<T>(error: unknown, create: () => T): ApiResult<T> {
  if (!MOCK_ENABLED) throw error
  if (error instanceof ApiError && error.status && error.status !== 404 && error.status < 500) {
    throw error
  }
  return { data: create(), mock: true, warning: mockWarning(error) }
}

export async function listAgents(): Promise<ApiResult<AgentProfile[]>> {
  try {
    const rawPayload = await request('/agents')
    const payload = asRecord(rawPayload)
    const agents = asArray(
      Array.isArray(rawPayload) ? rawPayload : (payload.agents ?? payload.items),
    ).map(normalizeAgent)
    if (!agents.length) throw new ApiError('API 未返回 Agent Profile')
    return { data: agents, mock: false }
  } catch (error) {
    return fallback(error, () => structuredClone(demoAgents))
  }
}

export async function createRun(
  input: CreateRunInput,
  useMock = false,
): Promise<ApiResult<RunRecord>> {
  if (useMock) {
    return {
      data: mockCreateRun(input),
      mock: true,
      warning: 'Agent 目录已进入显式演示模式；本 Run 不会发送到后端。',
    }
  }
  // Never turn an ambiguous POST network/5xx failure into a mock success: the
  // backend may already have durably accepted the run.
  const payload = await request('/runs', {
    method: 'POST',
    body: JSON.stringify({
      goal: input.task,
      agent_profile_id: input.agentId,
      skill_pack_id: input.skillPackId,
      execution_mode: input.executionMode,
    }),
  })
  const run = normalizeRun(payload)
  run.task = input.task
  run.skillPackId = input.skillPackId
  return { data: run, mock: false }
}

export async function getRun(id: string, useMock = false): Promise<ApiResult<RunRecord>> {
  if (useMock) {
    return {
      data: advanceMockRun(id),
      mock: true,
      warning: '本 Run 由前端演示运行器推进，未调用真实后端。',
    }
  }
  const payload = await request(`/runs/${encodeURIComponent(id)}`)
  return { data: normalizeRun(payload), mock: false }
}

export async function getRunJob(id: string, useMock = false): Promise<OperationJob> {
  if (useMock) return mockJob(id)
  return normalizeJob(await request(`/runs/${encodeURIComponent(id)}/job`))
}

export async function getRunAudit(id: string, useMock = false): Promise<AuditEvent[]> {
  if (useMock) return mockAudit(id)
  const payload = await request(`/runs/${encodeURIComponent(id)}/audit`)
  return asArray(payload).map(normalizeAuditEvent)
}

export async function getMetrics(id: string, useMock = false): Promise<MetricsSnapshot> {
  if (useMock) return mockMetrics(id)
  const payload = await request(`/metrics?run_id=${encodeURIComponent(id)}`)
  return normalizeMetrics(payload)
}

export async function listMcpServers(): Promise<McpServerSummary[]> {
  const payload = await request('/mcp/servers')
  return asArray(payload).map(normalizeMcpServer)
}

export async function decideApproval(
  id: string,
  approvalId: string,
  approved: boolean,
  useMock = false,
): Promise<ApiResult<RunRecord>> {
  if (useMock) {
    return {
      data: mockApprove(id, approved),
      mock: true,
      warning: '审批仅作用于前端演示状态，没有产生真实副作用。',
    }
  }
  const payload = await request(`/runs/${encodeURIComponent(id)}/approve`, {
    method: 'POST',
    body: JSON.stringify({
      call_id: approvalId,
      approved,
      reason: approved ? 'approved_in_workbench' : 'rejected_in_workbench',
    }),
  })
  return { data: normalizeRun(payload), mock: false }
}

export function createClientCommandKey(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  return `${prefix}-${suffix}`
}

export async function listReviewCases(): Promise<ReviewCaseList> {
  const payload = asRecord(await request('/review-cases'))
  return {
    items: asArray(payload.items).map(normalizeReviewCase),
    execution: normalizeReviewDisclosure(payload.execution),
  }
}

export async function getReviewCase(caseId: string): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}`),
  )
}

export async function createReviewCase(
  input: CreateReviewCaseInput,
  idempotencyKey: string,
): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request('/review-cases', {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        kind: input.kind,
        title: input.title,
        submission: {
          request_summary: input.requestSummary,
          business_justification: input.businessJustification,
          attributes: {},
          evidence_refs: [
            {
              evidence_id: input.evidenceId,
              source_type: 'document',
              locator: input.evidenceLocator,
              excerpt: input.evidenceExcerpt,
              title: input.evidenceId,
            },
          ],
        },
      }),
    }),
  )
}

export async function submitAndStartReviewCase(
  caseId: string,
  idempotencyKey: string,
): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}/submit-and-start`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
    }),
  )
}

export async function executeNextReviewRole(caseId: string): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}/execute-next`, {
      method: 'POST',
    }),
  )
}

export async function runReviewCaseUntilReview(caseId: string): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}/run-until-review`, {
      method: 'POST',
      body: JSON.stringify({ max_iterations: 4 }),
    }),
  )
}

export async function decideReviewRoleApproval(
  caseId: string,
  callId: string,
  approved: boolean,
): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}/role-approval`, {
      method: 'POST',
      body: JSON.stringify({
        call_id: callId,
        approved,
        reason: approved ? 'Approved in the review workbench.' : 'Denied in the review workbench.',
      }),
    }),
  )
}

export async function decideReviewCase(
  caseId: string,
  input: ReviewDecisionInput,
  idempotencyKey: string,
): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/review-cases/${encodeURIComponent(caseId)}/decision`, {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        expected_revision: input.expectedRevision,
        outcome: input.outcome,
        rationale: input.rationale,
        evidence_ref_ids: input.evidenceRefIds,
      }),
    }),
  )
}

function normalizePaperCard(value: unknown): PaperCard {
  const item = asRecord(value)
  return {
    paperId: text(item.paper_id),
    title: text(item.title ?? item.canonical_title, 'Untitled paper'),
    authors: stringArray(item.authors),
    abstract: text(item.abstract),
    shortDescription: text(item.short_description),
    year: numberValue(item.year),
    venue: text(item.venue) || undefined,
    doi: text(item.doi) || undefined,
    arxivId: text(item.arxiv_id) || undefined,
    sourceUrls: stringArray(item.source_urls),
    citationCount: numberValue(item.citation_count),
    relevanceScore: numberValue(item.relevance_score) ?? 0,
    relevanceReason: text(item.relevance_reason),
    verificationStatus: (text(item.verification_status, 'unverified')) as PaperCard['verificationStatus'],
    fullTextStatus: text(item.full_text_status, 'not_requested'),
  }
}

function normalizeProviderReport(value: unknown): LiteratureProviderReport {
  const item = asRecord(value)
  return {
    provider: text(item.provider, 'unknown'),
    queryCount: numberValue(item.query_count) ?? 0,
    resultCount: numberValue(item.result_count) ?? 0,
    requestCount: numberValue(item.request_count) ?? 0,
    cacheHits: numberValue(item.cache_hits) ?? 0,
    elapsedMs: numberValue(item.elapsed_ms) ?? 0,
    failure: text(item.failure) || undefined,
  }
}

function normalizeDiscovery(value: unknown): LiteratureDiscoveryResult {
  const item = asRecord(value)
  return {
    requestId: text(item.request_id),
    papers: asArray(item.papers).map(normalizePaperCard),
    providers: asArray(item.provider_reports).map(normalizeProviderReport),
    totalRawCandidates: numberValue(item.total_raw_candidates) ?? 0,
    queryRewriteApplied: Boolean(item.query_rewrite_applied),
  }
}

function normalizeResearchScope(value: unknown): ResearchScope {
  const item = asRecord(value)
  return {
    scopeId: text(item.scope_id),
    requestId: text(item.request_id),
    conversationId: text(item.conversation_id),
    selectedPaperIds: stringArray(item.selected_paper_ids),
    excludedPaperIds: stringArray(item.excluded_paper_ids),
    userIntent: text(item.user_intent),
    allowedExpansion: Boolean(item.allowed_expansion),
    scopeVersion: numberValue(item.scope_version) ?? 1,
    status: text(item.status),
    createdAt: text(item.created_at),
    confirmedAt: text(item.confirmed_at) || undefined,
  }
}

function normalizeIngestion(value: unknown): IngestionStatus {
  const item = asRecord(value)
  return {
    jobId: text(item.job_id),
    scopeId: text(item.scope_id),
    paperId: text(item.paper_id),
    status: text(item.status),
    evidenceCount: numberValue(item.evidence_count) ?? 0,
    error: text(item.error) || undefined,
    updatedAt: text(item.updated_at),
  }
}

function normalizeResearchEvidence(value: unknown): ResearchEvidenceCard {
  const item = asRecord(value)
  return {
    evidenceId: text(item.evidence_id),
    scopeId: text(item.scope_id) || undefined,
    scopeVersion: numberValue(item.scope_version),
    paperId: text(item.paper_id) || undefined,
    chunkId: text(item.chunk_id) || undefined,
    source: text(item.source),
    title: text(item.title) || undefined,
    section: text(item.section) || undefined,
    page: text(item.page) || undefined,
    evidenceType: text(item.evidence_type, 'paragraph'),
    snippet: text(item.snippet),
    score: numberValue(item.score) ?? 0,
    retrievalSources: stringArray(item.retrieval_sources),
    verificationStatus: text(item.verification_status, 'unread'),
  }
}

function normalizeConfidence(value: unknown): RetrievalConfidence {
  const item = asRecord(value)
  return {
    topScore: numberValue(item.top_score) ?? 0,
    queryTermCoverage: numberValue(item.query_term_coverage) ?? 0,
    sourceCoverage: numberValue(item.source_coverage) ?? 0,
    citationReadyCount: numberValue(item.citation_ready_count) ?? 0,
    scopePaperCoverage: numberValue(item.scope_paper_coverage) ?? 0,
    sufficient: Boolean(item.sufficient),
    reasons: stringArray(item.reasons),
  }
}

function normalizeScopeEvidence(value: unknown): ScopeEvidenceResult {
  const item = asRecord(value)
  return {
    scopeId: text(item.scope_id),
    scopeVersion: numberValue(item.scope_version) ?? 1,
    query: text(item.query),
    routedIntent: text(item.routed_intent),
    rewrittenQuery: text(item.rewritten_query) || undefined,
    retrievalRounds: numberValue(item.retrieval_rounds) ?? 1,
    activatedOperators: stringArray(item.activated_operators),
    evidence: asArray(item.evidence).map(normalizeResearchEvidence),
    confidence: normalizeConfidence(item.confidence),
  }
}

export async function searchLiterature(
  input: LiteratureSearchInput,
): Promise<LiteratureDiscoveryResult> {
  return normalizeDiscovery(
    await request('/literature/search', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: input.conversationId,
        request: {
          request_id: input.requestId,
          query: input.query,
          research_questions: input.researchQuestions,
          year_from: input.yearFrom,
          year_to: input.yearTo,
          required_terms: input.requiredTerms,
          excluded_terms: input.excludedTerms,
          result_limit: input.resultLimit,
        },
      }),
    }),
  )
}

export async function expandLiterature(
  requestId: string,
  seedPaperIds: string[],
): Promise<LiteratureDiscoveryResult> {
  return normalizeDiscovery(
    await request('/literature/expand-citations', {
      method: 'POST',
      body: JSON.stringify({
        request_id: requestId,
        seed_paper_ids: seedPaperIds,
        include_references: true,
        include_citations: true,
        per_seed_limit: 10,
        total_limit: 50,
      }),
    }),
  )
}

export async function createResearchScope(input: {
  requestId: string
  conversationId: string
  selectedPaperIds: string[]
  excludedPaperIds: string[]
  userIntent: string
  allowedExpansion: boolean
}): Promise<ResearchScope> {
  return normalizeResearchScope(
    await request('/research/scopes', {
      method: 'POST',
      body: JSON.stringify({
        request_id: input.requestId,
        conversation_id: input.conversationId,
        selected_paper_ids: input.selectedPaperIds,
        excluded_paper_ids: input.excludedPaperIds,
        user_intent: input.userIntent,
        allowed_expansion: input.allowedExpansion,
        confirm: true,
      }),
    }),
  )
}

export async function ingestResearchScope(scopeId: string): Promise<IngestionStatus[]> {
  return asArray(
    await request(`/research/scopes/${encodeURIComponent(scopeId)}/ingest`, {
      method: 'POST',
    }),
  ).map(normalizeIngestion)
}

export async function uploadResearchPaperPdf(
  scopeId: string,
  paperId: string,
  file: File,
): Promise<IngestionStatus> {
  return normalizeIngestion(
    await request(
      `/research/scopes/${encodeURIComponent(scopeId)}/papers/${encodeURIComponent(paperId)}/pdf`,
      {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/pdf',
          'X-Filename': encodeURIComponent(file.name),
        },
        body: file,
      },
    ),
  )
}

export async function uploadResearchPdfDirect(input: {
  conversationId: string
  userIntent: string
  title?: string
  file: File
}): Promise<{ scope: ResearchScope; paper: PaperCard; upload: IngestionStatus }> {
  const query = new URLSearchParams({
    conversation_id: input.conversationId,
    user_intent: input.userIntent,
  })
  if (input.title?.trim()) query.set('title', input.title.trim())
  const raw = asRecord(
    await request(`/research/uploads?${query.toString()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/pdf',
        'X-Filename': encodeURIComponent(input.file.name),
      },
      body: input.file,
    }),
  )
  return {
    scope: normalizeResearchScope(raw.scope),
    paper: normalizePaperCard(raw.paper),
    upload: normalizeIngestion(raw.upload),
  }
}

export async function getResearchScope(scopeId: string): Promise<ResearchScope> {
  return normalizeResearchScope(
    await request(`/research/scopes/${encodeURIComponent(scopeId)}`),
  )
}

export async function searchResearchEvidence(input: {
  scopeId: string
  scopeVersion: number
  query: string
  intent: string
}): Promise<ScopeEvidenceResult> {
  return normalizeScopeEvidence(
    await request('/research/evidence/search', {
      method: 'POST',
      body: JSON.stringify({
        scope_id: input.scopeId,
        scope_version: input.scopeVersion,
        query: input.query,
        intent: input.intent,
        top_k: 10,
        candidate_k: 50,
        mode: 'rigorous',
      }),
    }),
  )
}

export async function createResearchAgentRun(
  scopeId: string,
  title: string,
  context: string,
): Promise<ReviewCaseDetail> {
  return normalizeReviewDetail(
    await request(`/research/scopes/${encodeURIComponent(scopeId)}/agent-run`, {
      method: 'POST',
      headers: { 'Idempotency-Key': createClientCommandKey('research-agent-run') },
      body: JSON.stringify({ title, context, survey_depth: 'rigorous' }),
    }),
  )
}
