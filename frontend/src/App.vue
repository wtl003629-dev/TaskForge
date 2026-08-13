<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  createClientCommandKey,
  createResearchAgentRun,
  createResearchScope,
  createReviewCase,
  createRun,
  decideReviewCase,
  decideReviewRoleApproval,
  decideApproval,
  getMetrics,
  getResearchScope,
  getReviewCase,
  getRun,
  getRunAudit,
  getRunJob,
  listAgents,
  listMcpServers,
  listReviewCases,
  expandLiterature,
  ingestResearchScope,
  runReviewCaseUntilReview,
  searchLiterature,
  searchResearchEvidence,
  submitAndStartReviewCase,
  uploadResearchPdfDirect,
  uploadResearchPaperPdf,
} from './api'
import type {
  AgentProfile,
  AuditEvent,
  CreateReviewCaseInput,
  ExecutionMode,
  IngestionStatus,
  LiteratureDiscoveryResult,
  McpServerSummary,
  MetricsSnapshot,
  OperationJob,
  PaperCard,
  ReviewCase,
  ReviewCaseDetail,
  ReviewCaseKind,
  ReviewExecutionDisclosure,
  ReviewHandoff,
  ReviewPlanSlot,
  ReviewRoleRun,
  ResearchScope,
  RunRecord,
  ScopeEvidenceResult,
  SkillPack,
} from './types'

type WorkbenchMode = 'research' | 'agent' | 'review'

const activeMode = ref<WorkbenchMode>('research')

const agents = ref<AgentProfile[]>([])
const selectedAgentId = ref('')
const selectedSkillPackId = ref('')
const executionMode = ref<ExecutionMode>('inline')
const task = ref('调研企业知识库中与 Agent 权限治理相关的资料，生成一份带来源的摘要。')
const run = ref<RunRecord | null>(null)
const runMode = ref<ExecutionMode>('inline')
const job = ref<OperationJob | null>(null)
const audit = ref<AuditEvent[]>([])
const metrics = ref<MetricsSnapshot | null>(null)
const mcpServers = ref<McpServerSummary[]>([])
const loading = ref(false)
const approving = ref(false)
const telemetryLoading = ref(false)
const mcpLoading = ref(false)
const errorMessage = ref('')
const operationsMessage = ref('')
const mcpMessage = ref('')
const demoMode = ref(false)
const demoWarning = ref('')
let pollTimer: number | undefined

const literatureQuery = ref('agentic retrieval augmented generation for evidence-grounded research')
const researchQuestionsText = ref('How does the system keep evidence retrieval auditable?')
const yearFrom = ref<number | undefined>(2020)
const yearTo = ref<number | undefined>(new Date().getFullYear())
const requiredTermsText = ref('retrieval, evidence')
const excludedTermsText = ref('')
const literatureResult = ref<LiteratureDiscoveryResult | null>(null)
const selectedPaperIds = ref<string[]>([])
const researchScope = ref<ResearchScope | null>(null)
const ingestionStatuses = ref<IngestionStatus[]>([])
const uploadedPaperIds = ref<string[]>([])
const researchIntent = ref('Summarize the selected papers and compare their evidence retrieval methods.')
const allowExpansion = ref(true)
const evidenceQuery = ref('What evidence supports the main retrieval design choices?')
const evidenceIntent = ref('general_fact')
const scopeEvidence = ref<ScopeEvidenceResult | null>(null)
const researchAgentDetail = ref<ReviewCaseDetail | null>(null)
const researchLoading = ref(false)
const researchError = ref('')
const researchMessage = ref('')
const directUploadTitle = ref('')

const reviewCases = ref<ReviewCase[]>([])
const selectedCaseId = ref('')
const reviewDetail = ref<ReviewCaseDetail | null>(null)
const reviewExecution = ref<ReviewExecutionDisclosure | null>(null)
const reviewLoading = ref(false)
const reviewListLoading = ref(false)
const reviewError = ref('')
const reviewLoaded = ref(false)
const caseKind = ref<ReviewCaseKind>('enterprise_change')
const caseTitle = ref('支付服务集群迁移审查')
const caseSummary = ref('将支付服务迁移至新集群，并在变更窗口内完成流量切换。')
const caseJustification = ref('旧集群即将停止支持，需要迁移以保持服务连续性。')
const caseEvidenceId = ref('change-ticket-17')
const caseEvidenceLocator = ref('case://change-ticket-17')
const caseEvidenceExcerpt = ref('变更单已由服务负责人批准，包含回滚步骤、监控指标和计划变更窗口。')
const decisionRationale = ref('')
const selectedDecisionEvidence = ref<string[]>([])
let pendingCreateKey = ''
const pendingCommandKeys = new Map<string, string>()

const selectedAgent = computed(() =>
  agents.value.find((agent) => agent.id === selectedAgentId.value),
)
const selectedPapers = computed<PaperCard[]>(() => {
  const selected = new Set(selectedPaperIds.value)
  return (literatureResult.value?.papers ?? []).filter((paper) => selected.has(paper.paperId))
})
const researchPhase = computed(() => {
  if (researchAgentDetail.value) return 5
  if (scopeEvidence.value) return 4
  if (researchScope.value?.status === 'ready') return 3
  if (researchScope.value) return 2
  if (literatureResult.value) return 1
  return 0
})
const allSelectedPapersUploaded = computed(() =>
  selectedPaperIds.value.length > 0
  && selectedPaperIds.value.every((paperId) => uploadedPaperIds.value.includes(paperId)),
)
const skillPacks = computed<SkillPack[]>(() => selectedAgent.value?.skillPacks ?? [])
const selectedSkillPack = computed(() =>
  skillPacks.value.find((pack) => pack.id === selectedSkillPackId.value),
)
const isTerminal = computed(() =>
  run.value
    ? ['completed', 'failed', 'cancelled', 'step_limit'].includes(run.value.status)
    : false,
)
const recentAudit = computed(() => [...audit.value].reverse().slice(0, 12))
const recentReviewAudit = computed(() =>
  [...(reviewDetail.value?.auditEvents ?? [])].reverse().slice(0, 10),
)
const currentReviewCase = computed(() => reviewDetail.value?.case ?? null)
const caseNeedsEvidence = computed(() => caseKind.value !== 'research_survey')
const reviewIsResolved = computed(() =>
  currentReviewCase.value
    ? ['approved', 'rejected', 'failed'].includes(currentReviewCase.value.status)
    : false,
)
const reviewDisclosureText = computed(() => {
  const disclosure = reviewExecution.value ?? reviewDetail.value?.execution
  if (!disclosure) return '正在读取执行模式…'
  if (disclosure.businessE2eVerified) {
    return `业务端到端链路已有持久化验证记录 · Provider: ${disclosure.provider}`
  }
  if (disclosure.liveSmokeVerified) {
    return `真实模型 API 冒烟已有持久化验证记录，但业务 E2E 尚未验证 · Provider: ${disclosure.provider}`
  }
  if (disclosure.providerConfigured) {
    return `Provider 已配置（${disclosure.provider}），但没有真实 API 冒烟或业务 E2E 验证记录；已配置 ≠ 已实测`
  }
  if (disclosure.mode === 'offline-deterministic-demo') {
    return '离线确定性演示 · 未调用真实大模型 API'
  }
  return `测试 Provider（${disclosure.provider}）· 未调用真实大模型 API`
})

const reviewVerificationHeading = computed(() => {
  const disclosure = reviewExecution.value ?? reviewDetail.value?.execution
  if (!disclosure) return 'EXECUTION MODE'
  if (disclosure.businessE2eVerified) return 'BUSINESS E2E VERIFIED'
  if (disclosure.liveSmokeVerified) return 'LIVE SMOKE VERIFIED'
  if (disclosure.providerConfigured) return 'CONFIGURED / NOT VERIFIED'
  return 'OFFLINE / TEST'
})

const statusText: Record<string, string> = {
  pending: '待执行',
  step_limit: '达到步数上限',
  queued: '已排队',
  leased: 'Worker 执行中',
  running: '执行中',
  waiting_approval: '等待审批',
  completed: '已完成',
  dead_letter: '进入死信',
  failed: '失败',
  cancelled: '已取消',
  rejected: '已拒绝',
  draft: '草稿',
  submitted: '已提交',
  waiting_human_review: '等待人工复核',
  approved: '已批准',
  succeeded: '已完成',
  ready: '就绪',
  degraded: '降级完成',
  proposed: '待验证建议',
  verified: '已验证',
  approve: '建议批准',
  reject: '建议拒绝',
  escalate: '建议升级处理',
}

const roleText: Record<string, string> = {
  intake_analyst: '材料受理分析员',
  compliance_reviewer: '合规审查员',
  risk_reviewer: '风险评估员',
  decision_synthesizer: '决策汇总员',
  retrieval_planner: '检索规划员',
  source_evaluator: '来源甄别员',
  synthesis_writer: '综合综述员',
  critical_reviewer: '批判审查员',
}

function readableStatus(status: string): string {
  return statusText[status] ?? status
}

function formatPercent(value?: number): string {
  return value === undefined ? '—' : `${Math.round(value * 100)}%`
}

function formatDuration(value?: number): string {
  if (value === undefined) return '—'
  return value >= 1_000 ? `${(value / 1_000).toFixed(2)} s` : `${Math.round(value)} ms`
}

function formatTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }).format(date)
}

function mcpState(server: McpServerSummary): string {
  if (!server.enabled) return '已停用'
  return server.mountedTools.length ? '已挂载' : '已配置'
}

function readableRole(roleId: string): string {
  return roleText[roleId] ?? roleId
}

function roleRunForSlot(slot: ReviewPlanSlot): ReviewRoleRun | undefined {
  return reviewDetail.value?.roleRuns
    .filter((item) => item.slotId === slot.slotId)
    .sort((left, right) => right.attempt - left.attempt)[0]
}

function handoffSourceLabel(handoff: ReviewHandoff): string {
  const run = reviewDetail.value?.roleRuns.find(
    (item) => item.roleRunId === handoff.fromRoleRunId,
  )
  return run ? `${readableRole(run.roleId)} · attempt ${run.attempt}` : handoff.fromRoleRunId
}

function handoffTargetLabel(handoff: ReviewHandoff): string {
  const slot = reviewDetail.value?.plan?.slots.find(
    (item) => item.slotId === handoff.toSlotId,
  )
  return slot ? readableRole(slot.roleId) : handoff.toSlotId
}

function shortFactId(factId: string): string {
  return factId.length > 8 ? `${factId.slice(0, 8)}…` : factId
}

function displayValue(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function commandKey(scope: string): string {
  const existing = pendingCommandKeys.get(scope)
  if (existing) return existing
  const next = createClientCommandKey(scope)
  pendingCommandKeys.set(scope, next)
  return next
}

function clearCommandKey(scope: string): void {
  pendingCommandKeys.delete(scope)
}

function upsertReviewCase(next: ReviewCase): void {
  const index = reviewCases.value.findIndex((item) => item.caseId === next.caseId)
  if (index === -1) reviewCases.value.unshift(next)
  else reviewCases.value.splice(index, 1, next)
}

function applyReviewDetail(detail: ReviewCaseDetail): void {
  const changedCase = selectedCaseId.value !== detail.case.caseId
  reviewDetail.value = detail
  reviewExecution.value = detail.execution
  selectedCaseId.value = detail.case.caseId
  upsertReviewCase(detail.case)
  if (changedCase) {
    selectedDecisionEvidence.value = detail.case.submission.evidenceRefs.map(
      (item) => item.evidenceId,
    )
    decisionRationale.value = ''
  }
}

async function loadReviewInbox(selectFirst = true): Promise<void> {
  reviewListLoading.value = true
  reviewError.value = ''
  try {
    const result = await listReviewCases()
    reviewCases.value = result.items
    reviewExecution.value = result.execution
    reviewLoaded.value = true
    const targetId = selectedCaseId.value || (selectFirst ? result.items[0]?.caseId : '')
    if (targetId && result.items.some((item) => item.caseId === targetId)) {
      applyReviewDetail(await getReviewCase(targetId))
    } else if (!result.items.length) {
      selectedCaseId.value = ''
      reviewDetail.value = null
    }
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '无法读取企业审查列表。'
  } finally {
    reviewListLoading.value = false
  }
}

async function selectReviewCase(caseId: string, force = false): Promise<void> {
  if (!caseId || (!force && caseId === selectedCaseId.value && reviewDetail.value)) return
  reviewLoading.value = true
  reviewError.value = ''
  try {
    applyReviewDetail(await getReviewCase(caseId))
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '无法读取审查详情。'
  } finally {
    reviewLoading.value = false
  }
}

async function handleCreateReviewCase(): Promise<void> {
  const input: CreateReviewCaseInput = {
    kind: caseKind.value,
    title: caseTitle.value.trim(),
    requestSummary: caseSummary.value.trim(),
    businessJustification: caseJustification.value.trim(),
    evidenceId: caseEvidenceId.value.trim(),
    evidenceLocator: caseEvidenceLocator.value.trim(),
    evidenceExcerpt: caseEvidenceExcerpt.value.trim(),
  }
  const requiredValues = caseNeedsEvidence.value
    ? [
        input.title,
        input.requestSummary,
        input.businessJustification,
        input.evidenceId,
        input.evidenceLocator,
        input.evidenceExcerpt,
      ]
    : [input.title, input.requestSummary, input.businessJustification]
  if (requiredValues.some((value) => !value)) return

  reviewLoading.value = true
  reviewError.value = ''
  try {
    pendingCreateKey ||= createClientCommandKey('create-review')
    const detail = await createReviewCase(input, pendingCreateKey)
    pendingCreateKey = ''
    applyReviewDetail(detail)
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '创建审查草稿失败。'
  } finally {
    reviewLoading.value = false
  }
}

async function handleStartReview(): Promise<void> {
  const reviewCase = currentReviewCase.value
  if (!reviewCase || reviewCase.status !== 'draft') return
  const scope = `start-review-${reviewCase.caseId}`
  reviewLoading.value = true
  reviewError.value = ''
  try {
    applyReviewDetail(
      await submitAndStartReviewCase(reviewCase.caseId, commandKey(scope)),
    )
    clearCommandKey(scope)
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '提交并启动审查失败。'
  } finally {
    reviewLoading.value = false
  }
}

async function handleRunReview(): Promise<void> {
  const reviewCase = currentReviewCase.value
  if (!reviewCase || reviewCase.status !== 'running') return
  reviewLoading.value = true
  reviewError.value = ''
  try {
    applyReviewDetail(await runReviewCaseUntilReview(reviewCase.caseId))
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '四角色审查执行失败。'
  } finally {
    reviewLoading.value = false
  }
}

async function handleRoleApproval(item: ReviewRoleRun, approved: boolean): Promise<void> {
  const reviewCase = currentReviewCase.value
  if (!reviewCase || !item.pendingApprovalCallId) return
  reviewLoading.value = true
  reviewError.value = ''
  try {
    applyReviewDetail(
      await decideReviewRoleApproval(
        reviewCase.caseId,
        item.pendingApprovalCallId,
        approved,
      ),
    )
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '处理角色工具审批失败。'
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewDecision(outcome: 'approved' | 'rejected'): Promise<void> {
  const reviewCase = currentReviewCase.value
  const rationale = decisionRationale.value.trim()
  if (!reviewCase || reviewCase.status !== 'waiting_human_review' || !rationale) return
  const scope = `decision-${reviewCase.caseId}-${reviewCase.revision}-${outcome}`
  reviewLoading.value = true
  reviewError.value = ''
  try {
    applyReviewDetail(
      await decideReviewCase(
        reviewCase.caseId,
        {
          expectedRevision: reviewCase.revision,
          outcome,
          rationale,
          evidenceRefIds: [...selectedDecisionEvidence.value],
        },
        commandKey(scope),
      ),
    )
    clearCommandKey(scope)
  } catch (error) {
    reviewError.value = error instanceof Error ? error.message : '提交人工决定失败。'
  } finally {
    reviewLoading.value = false
  }
}

function splitResearchTerms(value: string): string[] {
  return [...new Set(value.split(/[,，;；\n]/).map((item) => item.trim()).filter(Boolean))]
}

function togglePaper(paperId: string): void {
  if (researchScope.value) return
  selectedPaperIds.value = selectedPaperIds.value.includes(paperId)
    ? selectedPaperIds.value.filter((item) => item !== paperId)
    : [...selectedPaperIds.value, paperId]
}

function paperTitle(paperId?: string): string {
  if (!paperId) return 'Unknown paper'
  return literatureResult.value?.papers.find((paper) => paper.paperId === paperId)?.title
    ?? paperId
}

function researchProtocol(item: ReviewRoleRun): string {
  const payload = item.roleResult?.research_payload
  return payload && typeof payload === 'object' && 'protocol' in payload
    ? String(payload.protocol)
    : 'no structured payload'
}

async function handleLiteratureSearch(): Promise<void> {
  const query = literatureQuery.value.trim()
  if (!query) return
  researchLoading.value = true
  researchError.value = ''
  researchMessage.value = '正在同时查询 Semantic Scholar、OpenAlex 与 arXiv…'
  try {
    const result = await searchLiterature({
      requestId: createClientCommandKey('literature'),
      conversationId: createClientCommandKey('research-conversation'),
      query,
      researchQuestions: splitResearchTerms(researchQuestionsText.value),
      yearFrom: yearFrom.value,
      yearTo: yearTo.value,
      requiredTerms: splitResearchTerms(requiredTermsText.value),
      excludedTerms: splitResearchTerms(excludedTermsText.value),
      resultLimit: 30,
    })
    literatureResult.value = result
    selectedPaperIds.value = []
    researchScope.value = null
    ingestionStatuses.value = []
    uploadedPaperIds.value = []
    scopeEvidence.value = null
    researchAgentDetail.value = null
    researchMessage.value = `已从开放文献源汇总 ${result.totalRawCandidates} 条候选，去重后返回 ${result.papers.length} 篇。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '开放文献检索失败。'
    researchMessage.value = ''
  } finally {
    researchLoading.value = false
  }
}

async function handleCitationExpansion(): Promise<void> {
  if (!literatureResult.value || !selectedPaperIds.value.length) return
  researchLoading.value = true
  researchError.value = ''
  researchMessage.value = '正在沿已选论文的引用与被引关系扩展候选集…'
  try {
    const expanded = await expandLiterature(
      literatureResult.value.requestId,
      selectedPaperIds.value.slice(0, 20),
    )
    const merged = new Map(literatureResult.value.papers.map((paper) => [paper.paperId, paper]))
    expanded.papers.forEach((paper) => merged.set(paper.paperId, paper))
    literatureResult.value = {
      ...literatureResult.value,
      papers: [...merged.values()],
      providers: expanded.providers,
      totalRawCandidates: literatureResult.value.totalRawCandidates + expanded.totalRawCandidates,
    }
    researchMessage.value = `引用图扩展新增 ${expanded.papers.length} 篇候选；ResearchScope 尚未改变。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '引用扩展失败。'
    researchMessage.value = ''
  } finally {
    researchLoading.value = false
  }
}

async function handleConfirmScope(): Promise<void> {
  if (!literatureResult.value || !selectedPaperIds.value.length || !researchIntent.value.trim()) return
  researchLoading.value = true
  researchError.value = ''
  try {
    const selected = new Set(selectedPaperIds.value)
    researchScope.value = await createResearchScope({
      requestId: literatureResult.value.requestId,
      conversationId: createClientCommandKey('scope-conversation'),
      selectedPaperIds: [...selectedPaperIds.value],
      excludedPaperIds: literatureResult.value.papers
        .map((paper) => paper.paperId)
        .filter((paperId) => !selected.has(paperId)),
      userIntent: researchIntent.value.trim(),
      allowedExpansion: allowExpansion.value,
    })
    scopeEvidence.value = null
    researchAgentDetail.value = null
    researchMessage.value = `Scope v${researchScope.value.scopeVersion} 已由 Host 确认，边界包含 ${selected.size} 篇论文。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '确认研究边界失败。'
  } finally {
    researchLoading.value = false
  }
}

async function handleScopeIngestion(): Promise<void> {
  if (!researchScope.value) return
  researchLoading.value = true
  researchError.value = ''
  try {
    ingestionStatuses.value = await ingestResearchScope(researchScope.value.scopeId)
    researchScope.value = await getResearchScope(researchScope.value.scopeId)
    const indexed = ingestionStatuses.value.filter((item) => item.status === 'indexed').length
    researchMessage.value = `已将 ${indexed} 篇用户上传的 PDF 写入 Scope v${researchScope.value.scopeVersion} 的有界索引。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '论文获取或索引失败。'
  } finally {
    researchLoading.value = false
  }
}

async function handlePaperUpload(paperId: string, event: Event): Promise<void> {
  if (!researchScope.value) return
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  researchLoading.value = true
  researchError.value = ''
  try {
    const status = await uploadResearchPaperPdf(
      researchScope.value.scopeId,
      paperId,
      file,
    )
    ingestionStatuses.value = [
      ...ingestionStatuses.value.filter((item) => item.paperId !== paperId),
      status,
    ]
    uploadedPaperIds.value = [...new Set([...uploadedPaperIds.value, paperId])]
    researchMessage.value = `已接收用户上传的 PDF：${paperTitle(paperId)}。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : 'PDF 上传失败。'
  } finally {
    researchLoading.value = false
    input.value = ''
  }
}

async function handleDirectResearchUpload(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file || !researchIntent.value.trim()) return
  researchLoading.value = true
  researchError.value = ''
  try {
    const result = await uploadResearchPdfDirect({
      conversationId: createClientCommandKey('direct-upload-conversation'),
      userIntent: researchIntent.value.trim(),
      title: directUploadTitle.value.trim() || undefined,
      file,
    })
    literatureResult.value = {
      requestId: result.scope.requestId,
      papers: [result.paper],
      providers: [],
      totalRawCandidates: 0,
      queryRewriteApplied: false,
    }
    selectedPaperIds.value = [result.paper.paperId]
    researchScope.value = result.scope
    ingestionStatuses.value = [result.upload]
    uploadedPaperIds.value = [result.paper.paperId]
    scopeEvidence.value = null
    researchAgentDetail.value = null
    researchMessage.value = 'PDF 已直接上传并建立研究边界，可以开始解析和索引。'
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : 'PDF 直接上传失败。'
  } finally {
    researchLoading.value = false
    input.value = ''
  }
}

async function handleEvidenceSearch(): Promise<void> {
  if (!researchScope.value || researchScope.value.status !== 'ready' || !evidenceQuery.value.trim()) return
  researchLoading.value = true
  researchError.value = ''
  try {
    scopeEvidence.value = await searchResearchEvidence({
      scopeId: researchScope.value.scopeId,
      scopeVersion: researchScope.value.scopeVersion,
      query: evidenceQuery.value.trim(),
      intent: evidenceIntent.value,
    })
    researchMessage.value = `已完成 ${scopeEvidence.value.retrievalRounds} 轮有界检索，返回 ${scopeEvidence.value.evidence.length} 条可追溯证据。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : 'Scope 内证据检索失败。'
  } finally {
    researchLoading.value = false
  }
}

async function handleResearchAgents(): Promise<void> {
  if (!researchScope.value || researchScope.value.status !== 'ready') return
  researchLoading.value = true
  researchError.value = ''
  researchMessage.value = '四个角色正在按结构化协议协作…'
  try {
    const created = await createResearchAgentRun(
      researchScope.value.scopeId,
      `Research survey: ${literatureQuery.value.slice(0, 120)}`,
      'Use the Host-confirmed ResearchScope and exchange only structured evidence IDs.',
    )
    researchAgentDetail.value = await runReviewCaseUntilReview(created.case.caseId)
    researchMessage.value = `四 Agent 已运行至 ${researchAgentDetail.value.case.status}，所有角色轨迹均已持久化。`
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '四 Agent 研究流程失败。'
    researchMessage.value = ''
  } finally {
    researchLoading.value = false
  }
}

function stopPolling(): void {
  if (pollTimer !== undefined) {
    window.clearTimeout(pollTimer)
    pollTimer = undefined
  }
}

function schedulePoll(): void {
  stopPolling()
  if (
    !run.value
    || isTerminal.value
    || run.value.status === 'waiting_approval'
    || job.value?.status === 'dead_letter'
  ) return
  pollTimer = window.setTimeout(() => void refreshRun(), 1500)
}

function useResult<T>(result: { data: T; mock: boolean; warning?: string }): T {
  demoMode.value = result.mock
  demoWarning.value = result.warning ?? ''
  return result.data
}

async function loadAgents(): Promise<void> {
  loading.value = true
  errorMessage.value = ''
  try {
    const result = await listAgents()
    agents.value = useResult(result)
    selectedAgentId.value = agents.value[0]?.id ?? ''
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '无法加载 Agent Profile。'
  } finally {
    loading.value = false
  }
}

async function loadMcpStatus(): Promise<void> {
  mcpLoading.value = true
  mcpMessage.value = ''
  try {
    mcpServers.value = await listMcpServers()
  } catch (error) {
    mcpMessage.value = error instanceof Error ? error.message : '无法读取 MCP 状态。'
  } finally {
    mcpLoading.value = false
  }
}

async function refreshTelemetry(): Promise<void> {
  if (!run.value) return
  const runId = run.value.id
  telemetryLoading.value = true
  operationsMessage.value = ''
  try {
    const [nextAudit, nextMetrics] = await Promise.all([
      getRunAudit(runId, demoMode.value),
      getMetrics(runId, demoMode.value),
    ])
    if (run.value?.id !== runId) return
    audit.value = nextAudit
    metrics.value = nextMetrics
  } catch (error) {
    operationsMessage.value = error instanceof Error ? error.message : '无法读取审计与指标。'
  } finally {
    telemetryLoading.value = false
  }
}

async function refreshJob(runId: string): Promise<void> {
  if (runMode.value !== 'queued') return
  try {
    const nextJob = await getRunJob(runId, demoMode.value)
    if (run.value?.id === runId) job.value = nextJob
  } catch (error) {
    operationsMessage.value = error instanceof Error ? error.message : '无法读取队列状态。'
  }
}

async function submitRun(): Promise<void> {
  if (!task.value.trim() || !selectedAgentId.value || !selectedSkillPackId.value) return
  stopPolling()
  loading.value = true
  errorMessage.value = ''
  operationsMessage.value = ''
  job.value = null
  audit.value = []
  metrics.value = null
  const submittedMode = executionMode.value
  try {
    const result = await createRun({
      task: task.value.trim(),
      agentId: selectedAgentId.value,
      skillPackId: selectedSkillPackId.value,
      executionMode: submittedMode,
    }, demoMode.value)
    run.value = useResult(result)
    runMode.value = submittedMode
    await Promise.all([refreshJob(run.value.id), refreshTelemetry()])
    schedulePoll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '创建 Run 失败。'
  } finally {
    loading.value = false
  }
}

async function refreshRun(): Promise<void> {
  if (!run.value) return
  const runId = run.value.id
  try {
    const [result] = await Promise.all([
      getRun(runId, demoMode.value),
      refreshJob(runId),
    ])
    if (run.value?.id !== runId) return
    run.value = useResult(result)
    await refreshTelemetry()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '刷新 Run 失败。'
  } finally {
    schedulePoll()
  }
}

async function handleApproval(approved: boolean): Promise<void> {
  if (!run.value?.pendingApproval) return
  approving.value = true
  errorMessage.value = ''
  try {
    const result = await decideApproval(
      run.value.id,
      run.value.pendingApproval.id,
      approved,
      demoMode.value,
    )
    run.value = useResult(result)
    await refreshTelemetry()
    schedulePoll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '提交审批决定失败。'
  } finally {
    approving.value = false
  }
}

watch(selectedAgentId, () => {
  selectedSkillPackId.value = skillPacks.value[0]?.id ?? ''
})

watch(
  [
    caseKind,
    caseTitle,
    caseSummary,
    caseJustification,
    caseEvidenceId,
    caseEvidenceLocator,
    caseEvidenceExcerpt,
  ],
  () => { pendingCreateKey = '' },
)

watch(activeMode, (mode) => {
  if (mode === 'review' && !reviewLoaded.value) void loadReviewInbox()
})

onMounted(() => {
  void loadAgents()
  void loadMcpStatus()
})
onBeforeUnmount(stopPolling)
</script>

<template>
  <main class="page-shell">
    <header class="hero">
      <div>
        <div class="brand-mark" aria-hidden="true">TF</div>
        <p class="eyebrow">TASKFORGE / GENERAL AGENT RUNTIME</p>
        <h1>把任务执行过程<br /><span>变成可以检查的证据。</span></h1>
      </div>
      <div class="hero-note">
        <strong>Phase 2 工作台</strong>
        <p>支持同步执行与持久队列，并展示 Worker、审计、指标和人工审批状态。</p>
        <p>MCP 卡片只反映宿主报告的挂载结果，不展示端点或凭据配置。</p>
      </div>
    </header>

    <nav class="workbench-tabs" role="tablist" aria-label="工作台模式">
      <button
        type="button"
        role="tab"
        :aria-selected="activeMode === 'research'"
        :class="{ active: activeMode === 'research' }"
        @click="activeMode = 'research'"
      >
        <span>01</span> 论文研究 Agent
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeMode === 'agent'"
        :class="{ active: activeMode === 'agent' }"
        @click="activeMode = 'agent'"
      >
        <span>02</span> 通用运行
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeMode === 'review'"
        :class="{ active: activeMode === 'review' }"
        @click="activeMode = 'review'"
      >
        <span>03</span> 企业审查
      </button>
    </nav>

    <template v-if="activeMode === 'research'">
      <div v-if="researchError" class="notice notice-error" role="alert">
        <strong>研究流程失败</strong><span>{{ researchError }}</span>
      </div>
      <div v-if="researchMessage" class="notice research-notice" role="status">
        <strong>HOST STATUS</strong><span>{{ researchMessage }}</span>
      </div>

      <ol class="research-steps" aria-label="论文研究流程">
        <li
          v-for="(label, index) in ['提出需求', '选择论文', '确认边界', '有界检索', '四 Agent', '人工复核']"
          :key="label"
          :class="{ active: researchPhase === index, done: researchPhase > index }"
        >
          <span>{{ String(index + 1).padStart(2, '0') }}</span><strong>{{ label }}</strong>
        </li>
      </ol>

      <section class="research-workspace">
        <aside class="research-query-panel panel">
          <div class="section-heading">
            <span>DISCOVERY</span>
            <div><p>开放文献发现</p><h2>先找论文，再锁定边界</h2></div>
          </div>
          <section class="direct-upload-card">
            <strong>已有论文？直接上传 PDF</strong>
            <p>无需联网发现，上传后直接进入有界 RAG。</p>
            <label class="field">
              <span>论文标题（可选）</span>
              <input v-model="directUploadTitle" maxlength="2000" :disabled="researchLoading" placeholder="默认使用文件名" />
            </label>
            <label class="secondary-button direct-upload-button">
              选择 PDF
              <input type="file" accept="application/pdf,.pdf" :disabled="researchLoading || !researchIntent.trim()" @change="handleDirectResearchUpload" />
            </label>
          </section>
          <label class="field">
            <span>研究需求</span>
            <textarea v-model="literatureQuery" rows="4" maxlength="4000" :disabled="researchLoading" />
          </label>
          <label class="field">
            <span>研究问题（逗号分隔）</span>
            <textarea v-model="researchQuestionsText" rows="3" maxlength="2000" :disabled="researchLoading" />
          </label>
          <div class="research-inline-fields">
            <label class="field"><span>起始年份</span><input v-model.number="yearFrom" type="number" min="1000" max="3000" /></label>
            <label class="field"><span>结束年份</span><input v-model.number="yearTo" type="number" min="1000" max="3000" /></label>
          </div>
          <label class="field"><span>必须包含</span><input v-model="requiredTermsText" placeholder="retrieval, evidence" /></label>
          <label class="field"><span>排除词</span><input v-model="excludedTermsText" placeholder="可选" /></label>
          <button class="primary-button" :disabled="researchLoading || !literatureQuery.trim()" @click="handleLiteratureSearch">
            {{ researchLoading ? '执行中…' : '跨源检索论文' }} <span aria-hidden="true">→</span>
          </button>

          <section v-if="literatureResult" class="provider-health">
            <p>PROVIDER HEALTH</p>
            <article v-for="provider in literatureResult.providers" :key="provider.provider" :data-failed="Boolean(provider.failure)">
              <div><strong>{{ provider.provider }}</strong><span>{{ provider.failure ? 'DEGRADED' : 'OK' }}</span></div>
              <small>{{ provider.resultCount }} results · {{ Math.round(provider.elapsedMs) }} ms · {{ provider.cacheHits }} cache</small>
              <small v-if="provider.failure">{{ provider.failure }}</small>
            </article>
          </section>
        </aside>

        <section class="research-results panel">
          <div class="section-heading research-results-heading">
            <span>PAPERS</span>
            <div><p>去重、核验、可解释排序</p><h2>候选论文 {{ literatureResult?.papers.length ?? 0 }}</h2></div>
            <button class="secondary-button" :disabled="researchLoading || !selectedPaperIds.length || Boolean(researchScope)" @click="handleCitationExpansion">沿引用图扩展</button>
          </div>

          <div v-if="!literatureResult" class="research-empty">
            <strong>输入研究需求开始</strong>
            <p>系统会并行查询三个真实学术数据源。开放发现阶段只负责推荐 PaperCard，不会混入后续段落召回率。</p>
          </div>
          <div v-else-if="!literatureResult.papers.length" class="research-empty">
            <strong>没有满足条件的论文</strong><p>请放宽年份、必含词或调整研究问题。</p>
          </div>
          <div v-else class="paper-list">
            <article
              v-for="paper in literatureResult.papers"
              :key="paper.paperId"
              class="paper-card"
              :class="{ selected: selectedPaperIds.includes(paper.paperId) }"
            >
              <button class="paper-select" :disabled="Boolean(researchScope)" :aria-pressed="selectedPaperIds.includes(paper.paperId)" @click="togglePaper(paper.paperId)">
                {{ selectedPaperIds.includes(paper.paperId) ? '✓ 已选择' : '+ 选择' }}
              </button>
              <div class="paper-score"><strong>{{ Math.round(paper.relevanceScore * 100) }}</strong><span>RELEVANCE</span></div>
              <div class="paper-copy">
                <div class="paper-badges">
                  <span :data-verification="paper.verificationStatus">{{ paper.verificationStatus }}</span>
                  <span>{{ paper.fullTextStatus }}</span><span v-if="paper.year">{{ paper.year }}</span>
                </div>
                <h3>{{ paper.title }}</h3>
                <p class="paper-authors">{{ paper.authors.slice(0, 5).join(', ') }}<template v-if="paper.venue"> · {{ paper.venue }}</template></p>
                <p class="paper-abstract">{{ paper.shortDescription || '暂无可验证的简短介绍。' }}</p>
                <div class="paper-links">
                  <a v-for="source in paper.sourceUrls.slice(0, 3)" :key="source" :href="source" target="_blank" rel="noreferrer">来源 ↗</a>
                  <span v-if="paper.doi">DOI {{ paper.doi }}</span><span v-if="paper.citationCount !== undefined">被引 {{ paper.citationCount }}</span>
                </div>
              </div>
            </article>
          </div>
        </section>

        <aside class="scope-panel panel">
          <div class="section-heading">
            <span>SCOPE</span><div><p>Host 权威边界</p><h2>ResearchScope</h2></div>
          </div>
          <template v-if="!researchScope">
            <p class="scope-summary">已选择 <strong>{{ selectedPaperIds.length }}</strong> 篇论文。确认后生成不可静默扩张、可版本审计的检索边界。</p>
            <ol class="selected-paper-list"><li v-for="paper in selectedPapers" :key="paper.paperId">{{ paper.title }}</li></ol>
            <label class="field"><span>研究意图</span><textarea v-model="researchIntent" rows="5" maxlength="4000" /></label>
            <label class="scope-checkbox"><input v-model="allowExpansion" type="checkbox" /><span>允许 Agent 提出扩界请求（仍需用户批准）</span></label>
            <button class="primary-button" :disabled="researchLoading || !selectedPaperIds.length || !researchIntent.trim()" @click="handleConfirmScope">确认研究边界</button>
          </template>
          <template v-else>
            <div class="scope-seal"><span>HOST CONFIRMED</span><strong>v{{ researchScope.scopeVersion }}</strong><small>{{ researchScope.status }}</small></div>
            <dl class="scope-facts">
              <div><dt>Scope ID</dt><dd><code>{{ researchScope.scopeId }}</code></dd></div>
              <div><dt>论文</dt><dd>{{ researchScope.selectedPaperIds.length }}</dd></div>
              <div><dt>扩界</dt><dd>{{ researchScope.allowedExpansion ? '仅请求 + 人工批准' : '禁止' }}</dd></div>
            </dl>
            <p class="scope-intent">{{ researchScope.userIntent }}</p>
            <section v-if="researchScope.status !== 'ready'" class="paper-upload-list">
              <p>请从候选链接自行下载 PDF，再逐篇上传。未上传的论文不会进入 RAG。</p>
              <label v-for="paper in selectedPapers" :key="paper.paperId" class="paper-upload-row">
                <span>{{ uploadedPaperIds.includes(paper.paperId) ? '✓ 已上传' : '选择 PDF' }}</span>
                <strong>{{ paper.title }}</strong>
                <input type="file" accept="application/pdf,.pdf" :disabled="researchLoading || uploadedPaperIds.includes(paper.paperId)" @change="handlePaperUpload(paper.paperId, $event)" />
              </label>
            </section>
            <button v-if="researchScope.status !== 'ready'" class="primary-button" :disabled="researchLoading || !allSelectedPapersUploaded" @click="handleScopeIngestion">解析已上传 PDF 并建立索引</button>
            <div v-if="ingestionStatuses.length" class="ingestion-list">
              <article v-for="item in ingestionStatuses" :key="item.jobId">
                <span :data-status="item.status">{{ item.status }}</span><strong>{{ paperTitle(item.paperId) }}</strong><small>{{ item.evidenceCount }} evidence</small>
              </article>
            </div>
          </template>
        </aside>
      </section>

      <section v-if="researchScope?.status === 'ready'" class="bounded-stage panel">
        <div class="bounded-stage-heading">
          <div><p>BOUND RETRIEVAL</p><h2>所有证据检索强制绑定 Scope v{{ researchScope.scopeVersion }}</h2></div>
          <code>{{ researchScope.scopeId }}</code>
        </div>
        <div class="bounded-controls">
          <label class="field"><span>在已选论文内提问</span><textarea v-model="evidenceQuery" rows="3" maxlength="4000" /></label>
          <label class="field"><span>检索意图</span>
            <select v-model="evidenceIntent">
              <option value="general_fact">一般事实</option><option value="method_definition">方法定义</option>
              <option value="experimental_setup">实验设置</option><option value="numeric_table">数值/表格</option>
              <option value="cross_paper_comparison">跨论文比较</option><option value="claim_verification">论断核验</option>
              <option value="related_work">相关工作</option>
            </select>
          </label>
          <button class="primary-button" :disabled="researchLoading || !evidenceQuery.trim()" @click="handleEvidenceSearch">运行有界检索</button>
        </div>

        <div v-if="scopeEvidence" class="evidence-stage">
          <section class="confidence-card" :data-sufficient="scopeEvidence.confidence.sufficient">
            <div><p>CONFIDENCE</p><strong>{{ scopeEvidence.confidence.sufficient ? 'SUFFICIENT' : 'NEEDS REVIEW' }}</strong></div>
            <dl><div><dt>轮次</dt><dd>{{ scopeEvidence.retrievalRounds }} / 2</dd></div><div><dt>Query 覆盖</dt><dd>{{ formatPercent(scopeEvidence.confidence.queryTermCoverage) }}</dd></div><div><dt>Scope 覆盖</dt><dd>{{ formatPercent(scopeEvidence.confidence.scopePaperCoverage) }}</dd></div><div><dt>可引用证据</dt><dd>{{ scopeEvidence.confidence.citationReadyCount }}</dd></div></dl>
            <p v-if="scopeEvidence.rewrittenQuery">低置信度触发针对性改写：{{ scopeEvidence.rewrittenQuery }}</p>
            <ul v-if="scopeEvidence.confidence.reasons.length"><li v-for="reason in scopeEvidence.confidence.reasons" :key="reason">{{ reason }}</li></ul>
          </section>
          <div class="research-evidence-list">
            <article v-for="item in scopeEvidence.evidence" :key="item.evidenceId">
              <header><span>{{ item.section || item.evidenceType }}</span><strong>{{ item.score.toFixed(4) }}</strong></header>
              <h3>{{ item.title || paperTitle(item.paperId) }}</h3><p>{{ item.snippet }}</p>
              <footer><code>{{ item.evidenceId }}</code><span>{{ item.page ? `page ${item.page}` : 'page n/a' }}</span><span>{{ item.retrievalSources.join(' + ') }}</span></footer>
            </article>
          </div>
        </div>

        <div class="agent-launch">
          <div><p>STRUCTURED MULTI-AGENT</p><h2>Planner → Evaluator → Writer → Critic</h2><span>角色之间只传计划、Evidence ID、ClaimManifest 和 ReviewPatch，不传完整聊天记录。</span></div>
          <button class="approve-button" :disabled="researchLoading" @click="handleResearchAgents">运行四 Agent</button>
        </div>
        <div v-if="researchAgentDetail" class="research-agent-grid">
          <article v-for="(item, index) in researchAgentDetail.roleRuns" :key="item.roleRunId">
            <span>{{ String(index + 1).padStart(2, '0') }}</span><h3>{{ readableRole(item.roleId) }}</h3>
            <strong :data-status="item.status">{{ readableStatus(item.status) }}</strong><p>{{ item.summary }}</p>
            <code>{{ researchProtocol(item) }}</code>
            <dl v-if="item.runtimeMetrics"><div><dt>Tool</dt><dd>{{ item.runtimeMetrics.toolSuccessCount }}/{{ item.runtimeMetrics.toolResultCount }}</dd></div><div><dt>耗时</dt><dd>{{ formatDuration(item.runtimeMetrics.elapsedMs) }}</dd></div><div><dt>证据</dt><dd>{{ item.retrievedEvidenceRefs.length }}</dd></div></dl>
          </article>
        </div>
      </section>
    </template>

    <div v-if="activeMode === 'agent' && demoMode" class="notice notice-demo" role="status">
      <strong>演示回退模式</strong>
      <span>{{ demoWarning || '当前数据来自本地 mock，不代表任务已真实执行。' }}</span>
    </div>
    <div v-if="activeMode === 'agent' && errorMessage" class="notice notice-error" role="alert">
      <strong>请求失败</strong><span>{{ errorMessage }}</span>
    </div>

    <section v-if="activeMode === 'agent'" class="workspace-grid">
      <aside class="control-panel panel">
        <div class="section-heading">
          <span>01</span>
          <div>
            <p>任务配置</p>
            <h2>选择能力边界</h2>
          </div>
        </div>

        <label class="field">
          <span>Agent Profile</span>
          <select v-model="selectedAgentId" :disabled="loading">
            <option v-for="agent in agents" :key="agent.id" :value="agent.id">
              {{ agent.name }}
            </option>
          </select>
        </label>
        <p class="field-hint">{{ selectedAgent?.description || '正在加载 Profile…' }}</p>

        <label class="field">
          <span>Skill Pack</span>
          <select v-model="selectedSkillPackId" :disabled="loading || !skillPacks.length">
            <option v-for="pack in skillPacks" :key="pack.id" :value="pack.id">
              {{ pack.name }}
            </option>
          </select>
        </label>
        <p class="field-hint">{{ selectedSkillPack?.description || '该 Profile 暂无可用 Skill。' }}</p>

        <div v-if="selectedSkillPack?.tools.length" class="capability-list">
          <span v-for="tool in selectedSkillPack.tools" :key="tool">{{ tool }}</span>
        </div>

        <fieldset class="mode-field">
          <legend>执行方式</legend>
          <label :class="{ selected: executionMode === 'inline' }">
            <input v-model="executionMode" type="radio" value="inline" />
            <span><strong>Inline</strong><small>请求内执行，立即返回状态</small></span>
          </label>
          <label :class="{ selected: executionMode === 'queued' }">
            <input v-model="executionMode" type="radio" value="queued" />
            <span><strong>Queued</strong><small>持久入队，由 Worker 租约执行</small></span>
          </label>
        </fieldset>

        <section class="mcp-card" aria-labelledby="mcp-title">
          <div class="compact-heading">
            <div>
              <span>MCP</span>
              <strong id="mcp-title">宿主挂载状态</strong>
            </div>
            <button type="button" :disabled="mcpLoading" @click="loadMcpStatus">
              {{ mcpLoading ? '读取中' : '刷新' }}
            </button>
          </div>
          <p v-if="mcpMessage" class="compact-message">状态暂不可用：{{ mcpMessage }}</p>
          <p v-else-if="!mcpServers.length" class="compact-message">未配置可展示的 MCP Server。</p>
          <ul v-else class="mcp-list">
            <li v-for="server in mcpServers" :key="server.namespace">
              <div>
                <code>{{ server.namespace }}</code>
                <span :data-state="mcpState(server)">{{ mcpState(server) }}</span>
              </div>
              <small>
                {{ server.mountedTools.length }} / {{ server.configuredTools.length }} tools ·
                {{ server.profileIds.length }} profiles
              </small>
            </li>
          </ul>
        </section>

        <label class="field task-field">
          <span>任务目标</span>
          <textarea
            v-model="task"
            rows="6"
            maxlength="2000"
            placeholder="描述目标、边界和预期交付物…"
          />
          <small>{{ task.length }} / 2000</small>
        </label>

        <button
          class="primary-button"
          :disabled="loading || !task.trim() || !selectedSkillPackId"
          @click="submitRun"
        >
          {{ loading ? '正在连接…' : '创建 Run' }}
          <span aria-hidden="true">↗</span>
        </button>
      </aside>

      <section class="run-panel panel">
        <div class="section-heading">
          <span>02</span>
          <div>
            <p>执行轨迹</p>
            <h2>Run Inspector</h2>
          </div>
          <div v-if="run" class="status-badge" :data-status="run.status">
            <i />{{ readableStatus(run.status) }}
          </div>
        </div>

        <div v-if="!run" class="empty-state">
          <div class="empty-orbit" aria-hidden="true"><span /></div>
          <h3>还没有运行记录</h3>
          <p>提交任务后，这里会显示状态、步骤、工具调用、证据和审批请求。</p>
        </div>

        <template v-else>
          <div class="run-meta">
            <div><span>RUN ID</span><code>{{ run.id }}</code></div>
            <div><span>PROFILE</span><strong>{{ run.agentId || selectedAgent?.name }}</strong></div>
            <div><span>SKILL</span><strong>{{ run.skillPackId || selectedSkillPack?.name }}</strong></div>
            <div><span>MODE</span><strong>{{ runMode.toUpperCase() }}</strong></div>
          </div>

          <section v-if="runMode === 'queued'" class="job-card" aria-labelledby="job-title">
            <div>
              <p>WORKER JOB</p>
              <h3 id="job-title">Durable execution</h3>
            </div>
            <template v-if="job">
              <span class="job-status" :data-status="job.status">
                <i />{{ readableStatus(job.status) }}
              </span>
              <dl>
                <div><dt>尝试</dt><dd>{{ job.attempt }} / {{ job.maxAttempts }}</dd></div>
                <div><dt>结果</dt><dd>{{ job.resultStatus ? readableStatus(job.resultStatus) : '—' }}</dd></div>
                <div><dt>更新时间</dt><dd>{{ formatTime(job.updatedAt) }}</dd></div>
                <div v-if="job.leaseExpiresAt">
                  <dt>租约到期</dt><dd>{{ formatTime(job.leaseExpiresAt) }}</dd>
                </div>
              </dl>
              <p v-if="job.lastError" class="job-error">{{ job.lastError }}</p>
            </template>
            <p v-else class="compact-message">正在读取持久队列状态…</p>
          </section>

          <div v-if="run.pendingApproval" class="approval-card">
            <div class="approval-icon" aria-hidden="true">!</div>
            <div class="approval-copy">
              <p>PENDING APPROVAL</p>
              <h3>{{ run.pendingApproval.toolName }}</h3>
              <p>{{ run.pendingApproval.reason }}</p>
              <dl>
                <div><dt>风险</dt><dd>{{ run.pendingApproval.risk }}</dd></div>
                <div v-if="run.pendingApproval.arguments">
                  <dt>参数</dt><dd><code>{{ JSON.stringify(run.pendingApproval.arguments) }}</code></dd>
                </div>
              </dl>
            </div>
            <div class="approval-actions">
              <button :disabled="approving" class="secondary-button" @click="handleApproval(false)">
                拒绝
              </button>
              <button :disabled="approving" class="approve-button" @click="handleApproval(true)">
                {{ approving ? '提交中…' : '确认执行' }}
              </button>
            </div>
          </div>

          <div class="steps">
            <article v-for="(step, index) in run.steps" :key="step.id" class="step-card">
              <div class="step-rail">
                <span>{{ String(index + 1).padStart(2, '0') }}</span>
                <i :data-status="step.status" />
              </div>
              <div class="step-content">
                <div class="step-title">
                  <div>
                    <h3>{{ step.title }}</h3>
                    <p v-if="step.summary">{{ step.summary }}</p>
                  </div>
                  <span class="mini-status" :data-status="step.status">
                    {{ readableStatus(step.status) }}
                  </span>
                </div>

                <div v-if="step.toolCalls.length" class="tool-list">
                  <div v-for="call in step.toolCalls" :key="call.id" class="tool-call">
                    <div>
                      <span class="tool-label">TOOL</span>
                      <code>{{ call.name }}</code>
                      <span class="tool-status">{{ readableStatus(call.status) }}</span>
                    </div>
                    <pre v-if="call.arguments">{{ JSON.stringify(call.arguments, null, 2) }}</pre>
                    <p v-if="call.result">{{ call.result }}</p>
                  </div>
                </div>
              </div>
            </article>
          </div>

          <section v-if="run.evidence.length || run.summary" class="evidence-section">
            <div class="evidence-heading">
              <div>
                <p>OUTPUT</p>
                <h3>Evidence &amp; Final State</h3>
              </div>
              <span>{{ run.evidence.length }} ITEMS</span>
            </div>
            <p v-if="run.summary" class="run-summary">{{ run.summary }}</p>
            <div class="evidence-grid">
              <article v-for="item in run.evidence" :key="item.id">
                <span>{{ item.kind }}</span>
                <h4>{{ item.title }}</h4>
                <p>{{ item.summary }}</p>
                <code v-if="item.source">{{ item.source }}</code>
              </article>
            </div>
          </section>

          <section class="observability-section" aria-labelledby="observability-title">
            <div class="observability-heading">
              <div>
                <p>OPERATIONS</p>
                <h3 id="observability-title">Audit &amp; Metrics</h3>
              </div>
              <button type="button" :disabled="telemetryLoading" @click="refreshTelemetry">
                {{ telemetryLoading ? '刷新中…' : '刷新' }}
              </button>
            </div>
            <p v-if="operationsMessage" class="compact-message operation-error">
              运行状态可用，但运维数据暂不可用：{{ operationsMessage }}
            </p>

            <div v-if="metrics" class="metrics-grid">
              <article>
                <span>RUN SUCCESS</span>
                <strong>{{ formatPercent(metrics.runSuccessRate) }}</strong>
                <small>{{ metrics.runSuccessCount }} / {{ metrics.runCount }} runs</small>
              </article>
              <article>
                <span>TOOL SUCCESS</span>
                <strong>{{ formatPercent(metrics.toolSuccessRate) }}</strong>
                <small>{{ metrics.toolSuccessCount }} / {{ metrics.toolCount }} calls</small>
              </article>
              <article>
                <span>P95 LATENCY</span>
                <strong>{{ formatDuration(metrics.durationP95Ms) }}</strong>
                <small>P50 {{ formatDuration(metrics.durationP50Ms) }}</small>
              </article>
              <article>
                <span>TOKENS</span>
                <strong>{{ metrics.totalTokens ?? '—' }}</strong>
                <small>input {{ metrics.inputTokens ?? '—' }} · output {{ metrics.outputTokens ?? '—' }}</small>
              </article>
              <article :data-alert="metrics.safetyViolationCount > 0">
                <span>SAFETY EVENTS</span>
                <strong>{{ metrics.safetyViolationCount }}</strong>
                <small>deterministic audit flag</small>
              </article>
            </div>

            <div class="audit-list">
              <article v-for="event in recentAudit" :key="event.eventId">
                <time :datetime="event.occurredAt">{{ formatTime(event.occurredAt) }}</time>
                <div>
                  <strong>{{ event.action }}</strong>
                  <small v-if="event.tool">tool / {{ event.tool }}</small>
                  <small v-else-if="event.provider">provider / {{ event.provider }}</small>
                </div>
                <span :data-outcome="event.outcome">{{ readableStatus(event.outcome) }}</span>
                <code>{{ formatDuration(event.durationMs) }}</code>
              </article>
              <p v-if="!recentAudit.length && !operationsMessage" class="compact-message">
                当前 Run 还没有审计事件。
              </p>
            </div>
          </section>
        </template>
      </section>
    </section>

    <template v-if="activeMode === 'review'">
      <div
        class="notice execution-notice"
        :class="reviewExecution?.liveSmokeVerified ? 'notice-live' : 'notice-demo'"
        role="status"
      >
        <strong>{{ reviewVerificationHeading }}</strong>
        <span>
          {{ reviewDisclosureText }}。所有模型结论均标记为 model_untrusted，最终批准或拒绝只能由用户提交。
        </span>
      </div>
      <section
        v-if="reviewExecution"
        class="verification-matrix"
        aria-label="模型能力验证状态"
      >
        <article :data-verified="reviewExecution.providerConfigured">
          <span>PROVIDER CONFIG</span>
          <strong>{{ reviewExecution.providerConfigured ? '已配置' : '未配置' }}</strong>
          <small>仅表示运行参数完整</small>
        </article>
        <article :data-verified="reviewExecution.contractTestedMock">
          <span>MOCK CONTRACT</span>
          <strong>{{ reviewExecution.contractTestedMock ? '已测试' : '未证明' }}</strong>
          <small>模拟 HTTP / Provider 契约</small>
        </article>
        <article :data-verified="reviewExecution.liveSmokeVerified">
          <span>LIVE SMOKE</span>
          <strong>{{ reviewExecution.liveSmokeVerified ? '已验证' : '未验证' }}</strong>
          <small>需要持久化真实调用凭据</small>
        </article>
        <article :data-verified="reviewExecution.businessE2eVerified">
          <span>BUSINESS E2E</span>
          <strong>{{ reviewExecution.businessE2eVerified ? '已验证' : '未验证' }}</strong>
          <small>需要完整业务链路证据</small>
        </article>
      </section>
      <div v-if="reviewError" class="notice notice-error" role="alert">
        <strong>审查操作失败</strong><span>{{ reviewError }}</span>
      </div>

      <section class="workspace-grid review-workspace">
        <aside class="control-panel panel review-control">
          <div class="section-heading">
            <span>01</span>
            <div>
              <p>CASE INTAKE</p>
              <h2>创建审查草稿</h2>
            </div>
          </div>

          <form class="review-form" @submit.prevent="handleCreateReviewCase">
            <label class="field">
              <span>审查类型</span>
              <select v-model="caseKind" :disabled="reviewLoading">
                <option value="enterprise_change">企业变更审查</option>
                <option value="enterprise_admission">企业准入审查</option>
                <option value="research_survey">文献综述</option>
              </select>
              <p v-if="!caseNeedsEvidence" class="field-hint">
                文献综述：只需研究问题，证据来自真实检索到的文献。
              </p>
            </label>
            <label class="field">
              <span>标题</span>
              <input v-model="caseTitle" required maxlength="500" />
            </label>
            <label class="field">
              <span>申请摘要</span>
              <textarea v-model="caseSummary" required rows="4" maxlength="16000" />
            </label>
            <label class="field">
              <span>业务理由</span>
              <textarea v-model="caseJustification" required rows="4" maxlength="16000" />
            </label>
            <div class="form-pair">
              <label class="field">
                <span>证据 ID</span>
                <input v-model="caseEvidenceId" :required="caseNeedsEvidence" maxlength="240" />
              </label>
              <label class="field">
                <span>证据定位符</span>
                <input v-model="caseEvidenceLocator" :required="caseNeedsEvidence" maxlength="2048" />
              </label>
            </div>
            <label class="field">
              <span>证据摘录</span>
              <textarea
                v-model="caseEvidenceExcerpt"
                :required="caseNeedsEvidence"
                rows="4"
                maxlength="16000"
                placeholder="粘贴能支持审查结论的原文片段；该内容会进入本案件隔离的知识库。"
              />
            </label>
            <p class="authority-note">
              身份、所有权、固定四角色 DAG 与模型建议均由宿主管理，不能在此表单中提交或覆盖。
            </p>
            <button
              class="primary-button"
              type="submit"
              :disabled="reviewLoading || !caseTitle.trim() || !caseSummary.trim() || !caseJustification.trim() || (caseNeedsEvidence && (!caseEvidenceId.trim() || !caseEvidenceLocator.trim() || !caseEvidenceExcerpt.trim()))"
            >
              {{ reviewLoading ? '正在处理…' : '保存为草稿' }}
              <span aria-hidden="true">↗</span>
            </button>
          </form>

          <section class="case-inbox" aria-labelledby="case-inbox-title">
            <div class="compact-heading">
              <div>
                <span>OWNER INBOX</span>
                <strong id="case-inbox-title">我的审查事项</strong>
              </div>
              <button type="button" :disabled="reviewListLoading" @click="loadReviewInbox(false)">
                {{ reviewListLoading ? '读取中' : '刷新' }}
              </button>
            </div>
            <p v-if="!reviewCases.length && !reviewListLoading" class="compact-message">
              当前没有审查事项。先创建一份草稿。
            </p>
            <div v-else class="case-list">
              <button
                v-for="item in reviewCases"
                :key="item.caseId"
                type="button"
                :class="{ selected: item.caseId === selectedCaseId }"
                @click="selectReviewCase(item.caseId)"
              >
                <span>{{ item.kind === 'enterprise_change' ? 'CHANGE' : 'ADMISSION' }}</span>
                <strong>{{ item.title }}</strong>
                <small>{{ readableStatus(item.status) }} · rev {{ item.revision }}</small>
              </button>
            </div>
          </section>
        </aside>

        <section class="run-panel panel review-panel">
          <div class="section-heading">
            <span>02</span>
            <div>
              <p>CONTROLLED REVIEW</p>
              <h2>Enterprise Review Inspector</h2>
            </div>
            <div
              v-if="currentReviewCase"
              class="status-badge"
              :data-status="currentReviewCase.status"
            >
              <i />{{ readableStatus(currentReviewCase.status) }}
            </div>
          </div>

          <div v-if="!reviewDetail" class="empty-state review-empty">
            <div class="empty-orbit" aria-hidden="true"><span /></div>
            <h3>选择或创建一个企业审查事项</h3>
            <p>草稿提交后，宿主会创建固定四角色 DAG；模型只能给出建议，不能替代最终人工决定。</p>
          </div>

          <template v-else>
            <div class="run-meta review-meta">
              <div><span>CASE ID</span><code>{{ reviewDetail.case.caseId }}</code></div>
              <div><span>KIND</span><strong>{{ reviewDetail.case.kind === 'enterprise_change' ? '企业变更' : '企业准入' }}</strong></div>
              <div><span>REVISION</span><strong>{{ reviewDetail.case.revision }}</strong></div>
              <div><span>FINAL AUTHORITY</span><strong>HUMAN</strong></div>
            </div>

            <section class="case-overview">
              <div>
                <p>REVIEW SUBJECT</p>
                <h3>{{ reviewDetail.case.title }}</h3>
                <p>{{ reviewDetail.case.submission.requestSummary }}</p>
              </div>
              <dl>
                <div><dt>业务理由</dt><dd>{{ reviewDetail.case.submission.businessJustification }}</dd></div>
                <div>
                  <dt>提交证据</dt>
                  <dd>
                    <div
                      v-for="evidence in reviewDetail.case.submission.evidenceRefs"
                      :key="evidence.evidenceId"
                      class="submitted-evidence"
                    >
                      <code>{{ evidence.evidenceId }} · {{ evidence.locator }}</code>
                      <span>{{ evidence.excerpt }}</span>
                    </div>
                  </dd>
                </div>
              </dl>
            </section>

            <div class="case-actions">
              <button
                v-if="reviewDetail.case.status === 'draft'"
                class="approve-button"
                type="button"
                :disabled="reviewLoading"
                @click="handleStartReview"
              >
                {{ reviewLoading ? '正在启动…' : '提交并启动固定 DAG' }}
              </button>
              <button
                v-if="reviewDetail.case.status === 'running'"
                class="approve-button"
                type="button"
                :disabled="reviewLoading"
                @click="handleRunReview"
              >
                {{ reviewLoading ? '四角色执行中…' : '运行四角色至人工复核' }}
              </button>
              <button
                class="secondary-button"
                type="button"
                :disabled="reviewLoading"
                @click="selectReviewCase(reviewDetail.case.caseId, true)"
              >
                刷新状态
              </button>
              <span v-if="reviewIsResolved">该事项已进入不可逆的最终状态。</span>
            </div>

            <section v-if="reviewDetail.plan" class="review-section">
              <div class="review-section-heading">
                <div>
                  <p>HOST-OWNED DAG</p>
                  <h3>固定四角色审查链</h3>
                </div>
                <span>{{ readableStatus(reviewDetail.plan.status) }} · v{{ reviewDetail.plan.version }}</span>
              </div>
              <div class="dag-grid">
                <article
                  v-for="(slot, index) in reviewDetail.plan.slots"
                  :key="slot.slotId"
                  :data-status="roleRunForSlot(slot)?.status || 'pending'"
                >
                  <div class="dag-order">{{ String(index + 1).padStart(2, '0') }}</div>
                  <div>
                    <span>{{ slot.slotId }}</span>
                    <h4>{{ readableRole(slot.roleId) }}</h4>
                    <p>{{ slot.dependsOn.length ? `依赖：${slot.dependsOn.join('、')}` : '起始节点' }}</p>
                  </div>
                  <strong>{{ readableStatus(roleRunForSlot(slot)?.status || 'pending') }}</strong>
                </article>
              </div>
            </section>

            <section v-if="reviewDetail.roleRuns.length" class="review-section">
              <div class="review-section-heading">
                <div>
                  <p>ROLE RUNS</p>
                  <h3>角色执行记录</h3>
                </div>
                <span>{{ reviewDetail.roleRuns.length }} RUNS</span>
              </div>
              <div class="role-run-grid">
                <article v-for="item in reviewDetail.roleRuns" :key="item.roleRunId">
                  <header>
                    <div>
                      <span>{{ item.slotId }} · attempt {{ item.attempt }}</span>
                      <h4>{{ readableRole(item.roleId) }}</h4>
                    </div>
                    <strong :data-status="item.status">{{ readableStatus(item.status) }}</strong>
                  </header>
                  <p v-if="item.summary">{{ item.summary }}</p>
                  <p v-if="item.summaryAuthority" class="untrusted-label">
                    {{ item.summaryAuthority }} · 角色摘要不构成已验证事实
                  </p>
                  <p v-if="item.citations.length" class="citation-line">
                    引用：{{ item.citations.join(' · ') }}
                  </p>
                  <p v-if="item.retrievedEvidenceRefs.length" class="citation-line">
                    本次检索 receipt：{{ item.retrievedEvidenceRefs.join(' · ') }}
                  </p>
                  <p v-if="item.runtimeMetrics" class="citation-line">
                    运行指标：{{ item.runtimeMetrics.stepCount }} steps ·
                    {{ item.runtimeMetrics.toolSuccessCount }}/{{ item.runtimeMetrics.toolResultCount }} tools ok ·
                    {{ Math.round(item.runtimeMetrics.elapsedMs) }} ms ·
                    {{ item.runtimeMetrics.usage?.totalTokens ?? 'token n/a' }}
                  </p>
                  <div v-if="item.pendingApprovalCallId" class="approval-actions">
                    <button
                      class="approve-button"
                      type="button"
                      :disabled="reviewLoading"
                      @click="handleRoleApproval(item, true)"
                    >
                      批准本次角色工具
                    </button>
                    <button
                      class="deny-button"
                      type="button"
                      :disabled="reviewLoading"
                      @click="handleRoleApproval(item, false)"
                    >
                      拒绝本次角色工具
                    </button>
                  </div>
                  <details v-if="item.roleResult">
                    <summary>查看结构化角色结果</summary>
                    <pre>{{ displayValue(item.roleResult) }}</pre>
                  </details>
                  <p v-if="item.error" class="job-error">{{ item.error }}</p>
                </article>
              </div>
            </section>

            <section v-if="reviewDetail.sharedFacts.length" class="review-section">
              <div class="review-section-heading">
                <div>
                  <p>SHARED FACTS</p>
                  <h3>跨角色事实层</h3>
                </div>
                <span>PROPOSED ≠ VERIFIED</span>
              </div>
              <div class="fact-list">
                <article v-for="fact in reviewDetail.sharedFacts" :key="fact.factId">
                  <div>
                    <code>{{ fact.factKey }}</code>
                    <span :data-status="fact.status">{{ readableStatus(fact.status) }}</span>
                  </div>
                  <pre>{{ displayValue(fact.value) }}</pre>
                  <small>
                    authority: {{ fact.authority }} · v{{ fact.version }}
                    <template v-if="fact.verifierRef"> · verifier: {{ fact.verifierRef }}</template>
                  </small>
                </article>
              </div>
            </section>

            <section v-if="reviewDetail.handoffs.length" class="review-section">
              <div class="review-section-heading">
                <div>
                  <p>VERIFIED HANDOFFS</p>
                  <h3>跨角色事实交接</h3>
                </div>
                <span>{{ reviewDetail.handoffs.length }} HANDOFFS</span>
              </div>
              <div class="handoff-list">
                <article v-for="handoff in reviewDetail.handoffs" :key="handoff.handoffId">
                  <div class="handoff-route">
                    <code>{{ handoffSourceLabel(handoff) }}</code>
                    <span aria-hidden="true">→</span>
                    <code>{{ handoffTargetLabel(handoff) }}</code>
                    <small>{{ formatTime(handoff.createdAt) }}</small>
                  </div>
                  <p class="handoff-summary">{{ handoff.summary }}</p>
                  <div v-if="handoff.sharedFactIds.length" class="handoff-facts">
                    <span>携带已验证事实</span>
                    <code
                      v-for="factId in handoff.sharedFactIds"
                      :key="factId"
                      :title="factId"
                    >
                      {{ shortFactId(factId) }}
                    </code>
                  </div>
                </article>
              </div>
            </section>

            <section v-if="reviewDetail.case.recommendation" class="recommendation-card">
              <div class="recommendation-heading">
                <div>
                  <p>MODEL RECOMMENDATION</p>
                  <h3>{{ readableStatus(reviewDetail.case.recommendation.outcome) }}</h3>
                </div>
                <span>MODEL_UNTRUSTED · 非最终决定</span>
              </div>
              <p class="recommendation-summary">{{ reviewDetail.case.recommendation.summary }}</p>
              <p>{{ reviewDetail.case.recommendation.rationale }}</p>
              <dl>
                <div><dt>置信度</dt><dd>{{ formatPercent(reviewDetail.case.recommendation.confidence) }}</dd></div>
                <div><dt>模型标识</dt><dd>{{ reviewDetail.case.recommendation.modelId }}</dd></div>
                <div>
                  <dt>证据绑定</dt>
                  <dd>{{ reviewDetail.case.recommendation.evidenceRefs.map((item) => item.evidenceId).join('、') }}</dd>
                </div>
              </dl>
            </section>

            <section
              v-if="reviewDetail.case.status === 'waiting_human_review'"
              class="human-decision-card"
              aria-labelledby="human-decision-title"
            >
              <div class="approval-icon" aria-hidden="true">H</div>
              <div>
                <p>HUMAN AUTHORITY REQUIRED</p>
                <h3 id="human-decision-title">由你作出最终决定</h3>
                <p>模型建议仅供参考。提交时后端绑定当前用户身份，并以 revision 做并发校验。</p>
                <label class="field">
                  <span>决定理由（必填）</span>
                  <textarea
                    v-model="decisionRationale"
                    rows="4"
                    maxlength="16000"
                    placeholder="说明批准或拒绝的依据、风险接受范围与后续动作…"
                  />
                </label>
                <fieldset class="evidence-checklist">
                  <legend>纳入决定的证据</legend>
                  <label
                    v-for="evidence in reviewDetail.case.submission.evidenceRefs"
                    :key="evidence.evidenceId"
                  >
                    <input
                      v-model="selectedDecisionEvidence"
                      type="checkbox"
                      :value="evidence.evidenceId"
                    />
                    <span>{{ evidence.evidenceId }} · {{ evidence.locator }}</span>
                  </label>
                </fieldset>
                <div class="decision-actions">
                  <button
                    class="secondary-button"
                    type="button"
                    :disabled="reviewLoading || !decisionRationale.trim()"
                    @click="handleReviewDecision('rejected')"
                  >
                    最终拒绝
                  </button>
                  <button
                    class="approve-button"
                    type="button"
                    :disabled="reviewLoading || !decisionRationale.trim()"
                    @click="handleReviewDecision('approved')"
                  >
                    {{ reviewLoading ? '正在提交…' : '最终批准' }}
                  </button>
                </div>
              </div>
            </section>

            <section v-if="reviewDetail.case.humanDecision" class="human-result">
              <p>FINAL HUMAN DECISION</p>
              <h3>{{ readableStatus(reviewDetail.case.humanDecision.outcome) }}</h3>
              <p>{{ reviewDetail.case.humanDecision.rationale }}</p>
              <small>
                authority: human
                <template v-if="reviewDetail.case.humanDecision.actor.displayName">
                  · {{ reviewDetail.case.humanDecision.actor.displayName }}
                </template>
                · {{ formatTime(reviewDetail.case.humanDecision.decidedAt) }}
              </small>
            </section>

            <section v-if="reviewDetail.case.failure" class="notice notice-error">
              <strong>审查失败</strong><span>{{ reviewDetail.case.failure.reason }}</span>
            </section>

            <section class="review-section case-audit">
              <div class="review-section-heading">
                <div>
                  <p>APPEND-ONLY AUDIT</p>
                  <h3>案件状态审计</h3>
                </div>
                <span>{{ reviewDetail.auditEvents.length }} EVENTS</span>
              </div>
              <div class="audit-list">
                <article v-for="event in recentReviewAudit" :key="event.eventId">
                  <time :datetime="event.createdAt">{{ formatTime(event.createdAt) }}</time>
                  <div>
                    <strong>{{ event.eventType }}</strong>
                    <small>{{ event.fromStatus || '—' }} → {{ event.toStatus }}</small>
                  </div>
                  <span>{{ event.actorAuthority }}</span>
                  <code>rev {{ event.revision }}</code>
                </article>
              </div>
            </section>
          </template>
        </section>
      </section>
    </template>
  </main>
</template>
