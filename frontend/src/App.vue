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
  listResearchEvidence,
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
  LiteratureLanguagePreference,
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
  ResearchEvidenceCard,
  ResearchScope,
  RunRecord,
  ScopeEvidenceResult,
  SkillPack,
} from './types'

type WorkbenchMode = 'research' | 'agent' | 'review'
type ResearchWorkspaceView = 'sources' | 'ask' | 'report'

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

const literatureQuery = ref('')
const researchQuestionsText = ref('')
const yearFrom = ref<number | undefined>(2020)
const yearTo = ref<number | undefined>(new Date().getFullYear())
const literatureResultLimit = ref(50)
const languagePreference = ref<LiteratureLanguagePreference>('balanced')
const requiredTermsText = ref('')
const excludedTermsText = ref('')
const literatureResult = ref<LiteratureDiscoveryResult | null>(null)
const selectedPaperIds = ref<string[]>([])
const researchScope = ref<ResearchScope | null>(null)
const ingestionStatuses = ref<IngestionStatus[]>([])
const uploadedPaperIds = ref<string[]>([])
const researchIntent = ref('')
const allowExpansion = ref(true)
const evidenceQuery = ref('')
const evidenceIntent = ref('general_fact')
const scopeEvidence = ref<ScopeEvidenceResult | null>(null)
const showAllEvidence = ref(false)
const activeEvidence = ref<ScopeEvidenceResult['evidence'][number] | null>(null)
const researchAgentDetail = ref<ReviewCaseDetail | null>(null)
const reportQuestion = ref(literatureQuery.value)
const reportEvidence = ref<ResearchEvidenceCard[]>([])
const researchLoading = ref(false)
const reportGenerating = ref(false)
const researchError = ref('')
const researchMessage = ref('')
const directUploadTitle = ref('')
const researchWorkspaceView = ref<ResearchWorkspaceView>('sources')

const exampleResearchQueries = [
  'RAG 的最新技术方向与评测方法有哪些？',
  '大语言模型在医疗问答中的可靠性如何评估？',
  '多模态检索增强生成有哪些代表性研究？',
] as const

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
const chinesePaperCount = computed(() =>
  (literatureResult.value?.papers ?? []).filter(isChinesePaper).length,
)
const sortedEvidence = computed(() =>
  [...(scopeEvidence.value?.evidence ?? [])].sort((left, right) => right.score - left.score),
)
const visibleEvidence = computed(() =>
  showAllEvidence.value ? sortedEvidence.value : sortedEvidence.value.slice(0, 5),
)
const hasMoreEvidence = computed(() => sortedEvidence.value.length > 5)
const canRetryScopeIngestion = computed(() =>
  ingestionStatuses.value.some((item) => item.status === 'uploaded'),
)
const researchStep = computed(() => {
  if (researchScope.value?.status === 'ready') return 4
  if (researchScope.value) return 3
  if (literatureResult.value) return 2
  return 1
})
const researchSteps = [
  { number: 1, title: '提出问题', detail: '设置检索需求' },
  { number: 2, title: '选择论文', detail: '确认研究清单' },
  { number: 3, title: '获取全文', detail: '自动获取或上传 PDF' },
  { number: 4, title: '证据与报告', detail: '检索原文并生成草稿' },
] as const
const researchStepGuide = computed(() => ({
  1: '输入研究问题，其他筛选条件可稍后展开设置。',
  2: '从结果中选择论文，然后保存研究清单。',
  3: '等待开放论文自动索引；受限论文请按提示上传 PDF。',
  4: '可以搜索论文全文，也可以直接生成带引用的研究报告。',
})[researchStep.value])
const pendingIngestionCount = computed(() => {
  if (!researchScope.value) return 0
  const indexed = new Set(
    ingestionStatuses.value
      .filter((item) => item.status === 'indexed')
      .map((item) => item.paperId),
  )
  return researchScope.value.selectedPaperIds.filter((paperId) => !indexed.has(paperId)).length
})
const indexedPaperCount = computed(() =>
  ingestionStatuses.value.filter((item) => item.status === 'indexed').length,
)
const reportAvailabilityReason = computed(() => {
  if (!researchScope.value) return '保存论文清单后即可进入报告生成。'
  if (reportGenerating.value) return 'Planner、Evaluator、Writer、Critic 正在依次处理论文证据。'
  if (researchScope.value.status !== 'ready') {
    return pendingIngestionCount.value
      ? `还有 ${pendingIngestionCount.value} 篇论文未完成全文索引，完成后即可生成报告。`
      : '正在确认全文索引状态，请稍候。'
  }
  if (researchAgentDetail.value?.case.status === 'waiting_human_review') {
    return '报告草稿已生成，当前等待人工核对引用。'
  }
  if (researchError.value && researchAgentDetail.value?.case.status === 'failed') {
    return '上次生成失败，可以保留当前论文清单重新生成。'
  }
  return '论文全文已就绪，可以生成带原文引用的研究报告。'
})

function ingestionForPaper(paperId: string): IngestionStatus | undefined {
  return ingestionStatuses.value.find((item) => item.paperId === paperId)
}
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

function compactEvidenceSnippet(value: string, limit = 320): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  if (normalized.length <= limit) return normalized
  const head = normalized.slice(0, limit)
  const sentenceEnds = [...head.matchAll(/[。！？.!?；;]/g)]
    .map((match) => (match.index ?? -1) + match[0].length)
    .filter((index) => index > 0)
  const end = sentenceEnds.length >= 3
    ? sentenceEnds[2]
    : sentenceEnds.length
      ? sentenceEnds[sentenceEnds.length - 1]
      : limit
  return `${normalized.slice(0, end).trimEnd()}…`
}

function reportParagraphs(value?: string): string[] {
  return (value ?? '')
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
}

function writerSummary(detail: ReviewCaseDetail): string {
  return [...detail.roleRuns]
    .reverse()
    .find((run) => run.roleId === 'synthesis_writer' && run.status === 'succeeded')
    ?.summary ?? ''
}

const reportSources = computed(() => {
  const evidenceIds = researchAgentDetail.value?.researchAnswer?.evidenceIds ?? []
  return evidenceIds.map((evidenceId, index) => ({
    number: index + 1,
    evidenceId,
    card: reportEvidence.value.find((item) => item.evidenceId === evidenceId),
  }))
})

function reportSourceLabel(source: (typeof reportSources.value)[number]): string {
  if (!source.card) return `[${source.number}] 原文证据`
  const title = source.card.title || paperTitle(source.card.paperId)
  const location = [source.card.page ? `第 ${source.card.page} 页` : '', source.card.section || '']
    .filter(Boolean)
    .join(' · ')
  return `[${source.number}] ${title}${location ? ` · ${location}` : ''}`
}

type EvidenceHighlightSegment = { text: string; highlighted: boolean }

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function evidenceQueryTerms(value: string): string[] {
  const normalized = value.toLocaleLowerCase().replace(/[“”"'‘’。，、！？；：,.!?;:()[\]{}<>/\\|]+/g, ' ')
  const terms: string[] = normalized.match(/[a-z0-9][a-z0-9_-]*/g) ?? []
  for (const run of normalized.match(/[\u3400-\u9fff]+/g) ?? []) {
    if (run.length >= 2) terms.push(run)
    for (let index = 0; index < run.length - 1; index += 1) {
      terms.push(run.slice(index, index + 2))
    }
  }
  return [...new Set(terms.filter((term) => term.length >= 2))]
}

function highlightEvidenceText(value: string): EvidenceHighlightSegment[] {
  const text = compactEvidenceSnippet(value)
  const terms = evidenceQueryTerms(evidenceQuery.value)
  if (!terms.length) return [{ text, highlighted: false }]
  const matcher = new RegExp(terms.sort((left, right) => right.length - left.length).map(escapeRegExp).join('|'), 'giu')
  const segments: EvidenceHighlightSegment[] = []
  let cursor = 0
  for (const match of text.matchAll(matcher)) {
    const start = match.index ?? 0
    if (start > cursor) segments.push({ text: text.slice(cursor, start), highlighted: false })
    segments.push({ text: match[0], highlighted: true })
    cursor = start + match[0].length
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), highlighted: false })
  return segments.length ? segments : [{ text, highlighted: false }]
}

function openEvidenceDrawer(item: ScopeEvidenceResult['evidence'][number]): void {
  activeEvidence.value = item
}

function closeEvidenceDrawer(): void {
  activeEvidence.value = null
}

function handleEvidenceKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape' && activeEvidence.value) closeEvidenceDrawer()
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

function clearSelectedPapers(): void {
  if (!researchScope.value) selectedPaperIds.value = []
}

function useExampleQuery(query: string): void {
  literatureQuery.value = query
  researchIntent.value = query
  reportQuestion.value = query
}

function startNewResearch(): void {
  literatureQuery.value = ''
  researchQuestionsText.value = ''
  requiredTermsText.value = ''
  excludedTermsText.value = ''
  literatureResult.value = null
  selectedPaperIds.value = []
  researchScope.value = null
  ingestionStatuses.value = []
  uploadedPaperIds.value = []
  researchIntent.value = ''
  evidenceQuery.value = ''
  scopeEvidence.value = null
  showAllEvidence.value = false
  closeEvidenceDrawer()
  researchAgentDetail.value = null
  reportQuestion.value = ''
  reportEvidence.value = []
  researchError.value = ''
  researchMessage.value = ''
  directUploadTitle.value = ''
  researchWorkspaceView.value = 'sources'
}

function showResearchView(view: ResearchWorkspaceView): void {
  if (view !== 'sources' && researchScope.value?.status !== 'ready') return
  researchWorkspaceView.value = view
}

function verificationLabel(status: PaperCard['verificationStatus']): string {
  return {
    cross_source_verified: '多源核验',
    provider_verified: '单一元数据源',
    metadata_partial: '元数据待核对',
    unverified: '待核验',
  }[status]
}

function publicationTypeLabel(publicationType?: string): string | undefined {
  if (!publicationType) return undefined
  return {
    article: '期刊论文',
    'journal-article': '期刊论文',
    'proceedings-article': '会议论文',
    preprint: '预印本',
    'posted-content': '预印本',
    review: '综述',
    dissertation: '学位论文',
    'book-chapter': '书籍章节',
    'peer-review': '同行评议',
  }[publicationType.toLowerCase()] ?? publicationType
}

function paperTrustLabel(paper: PaperCard): string {
  if (paper.verificationStatus === 'cross_source_verified') return '多源学术核验'
  if (paper.verificationStatus === 'provider_verified' && (paper.venue || paper.publisher)) {
    return '来源可核对'
  }
  if (paper.venue || paper.publisher || paper.doi) return '出版信息可核对'
  return '来源信息有限'
}

function paperTrustTone(paper: PaperCard): string {
  if (paper.verificationStatus === 'cross_source_verified') return 'strong'
  if (paper.verificationStatus === 'provider_verified' || paper.venue || paper.publisher || paper.doi) {
    return 'standard'
  }
  return 'limited'
}

function fullTextLabel(status: string): string {
  return {
    available: '开放全文',
    ingested: '已建立索引',
    abstract_only: '仅摘要',
    failed: '需要上传',
    not_requested: '待获取全文',
  }[status] ?? status
}

function ingestionStatusLabel(status: string): string {
  return {
    queued: '等待处理',
    uploaded: '已上传，待解析',
    fetching: '正在获取全文',
    parsing: '正在解析',
    indexed: '已完成索引',
    abstract_only: '仅有摘要',
    failed: '需要手动上传',
  }[status] ?? status
}

function providerLabel(provider: string): string {
  return {
    semantic_scholar: 'Semantic Scholar',
    openalex: 'OpenAlex',
    arxiv: 'arXiv',
    crossref: 'Crossref',
  }[provider] ?? provider
}

function paperLanguageLabel(paper: PaperCard): string | undefined {
  if (paper.language?.toLowerCase().startsWith('zh')) return '中文'
  if (paper.language?.toLowerCase().startsWith('en')) return '英文'
  if (paper.language) return paper.language.toUpperCase()
  if (isChinesePaper(paper)) return '中文'
  return /[a-z]/iu.test(paper.title) ? '英文' : undefined
}

function isChinesePaper(paper: PaperCard): boolean {
  return paper.language?.toLowerCase().startsWith('zh') === true
    || /[\u3400-\u9fff]/u.test(paper.title)
}

function paperTitle(paperId?: string): string {
  if (!paperId) return 'Unknown paper'
  return literatureResult.value?.papers.find((paper) => paper.paperId === paperId)?.title
    ?? paperId
}

async function handleLiteratureSearch(): Promise<void> {
  const query = literatureQuery.value.trim()
  if (!query) return
  researchLoading.value = true
  researchError.value = ''
  researchMessage.value = '正在用中英文双路查询 Semantic Scholar、OpenAlex、arXiv 与 Crossref…'
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
      languagePreference: languagePreference.value,
      resultLimit: literatureResultLimit.value,
    })
    literatureResult.value = result
    selectedPaperIds.value = []
    researchScope.value = null
    ingestionStatuses.value = []
    uploadedPaperIds.value = []
    scopeEvidence.value = null
    showAllEvidence.value = false
    closeEvidenceDrawer()
    researchAgentDetail.value = null
    reportEvidence.value = []
    researchWorkspaceView.value = 'sources'
    researchIntent.value = query
    reportQuestion.value = query
    const chineseCount = result.papers.filter(isChinesePaper).length
    researchMessage.value = `中英文双路共汇总 ${result.totalRawCandidates} 条候选，返回 ${result.papers.length} 篇，其中中文论文 ${chineseCount} 篇。`
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
    researchMessage.value = `引用关系新增 ${expanded.papers.length} 篇候选，已选论文保持不变。`
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
    reportQuestion.value = researchIntent.value.trim()
    scopeEvidence.value = null
    showAllEvidence.value = false
    closeEvidenceDrawer()
    researchAgentDetail.value = null
    reportEvidence.value = []
    researchMessage.value = '论文清单已保存，正在自动获取可合法下载的开放 PDF…'
    ingestionStatuses.value = await ingestResearchScope(researchScope.value.scopeId)
    researchScope.value = await getResearchScope(researchScope.value.scopeId)
    researchWorkspaceView.value = researchScope.value.status === 'ready' ? 'ask' : 'sources'
    const indexed = ingestionStatuses.value.filter((item) => item.status === 'indexed').length
    const manual = ingestionStatuses.value.filter((item) => item.status === 'failed').length
    researchMessage.value = manual
      ? `已自动获取并索引 ${indexed} 篇；另有 ${manual} 篇受访问权限限制，请通过来源链接自行下载后上传。`
      : `已自动获取并索引全部 ${indexed} 篇开放论文。`
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
    if (researchScope.value.status === 'ready') researchWorkspaceView.value = 'ask'
    const indexed = ingestionStatuses.value.filter((item) => item.status === 'indexed').length
    const manual = ingestionStatuses.value.filter((item) => item.status === 'failed').length
    researchMessage.value = manual
      ? `已索引 ${indexed} 篇；仍有 ${manual} 篇需要通过来源链接自行下载后上传。`
      : `已将 ${indexed} 篇 PDF 写入当前论文清单的有界索引。`
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
  const uploadIntent = researchIntent.value.trim() || literatureQuery.value.trim()
  if (!file || !uploadIntent) return
  researchLoading.value = true
  researchError.value = ''
  try {
    const result = await uploadResearchPdfDirect({
      conversationId: createClientCommandKey('direct-upload-conversation'),
      userIntent: uploadIntent,
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
    showAllEvidence.value = false
    closeEvidenceDrawer()
    researchAgentDetail.value = null
    researchIntent.value = uploadIntent
    reportQuestion.value = uploadIntent
    reportEvidence.value = []
    researchWorkspaceView.value = 'sources'
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
  researchWorkspaceView.value = 'ask'
  researchError.value = ''
  showAllEvidence.value = false
  closeEvidenceDrawer()
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
  const question = reportQuestion.value.trim()
  if (!researchScope.value || researchScope.value.status !== 'ready' || !question) return
  researchLoading.value = true
  researchWorkspaceView.value = 'report'
  reportGenerating.value = true
  researchError.value = ''
  reportEvidence.value = []
  researchMessage.value = '四个角色正在按结构化协议协作…'
  try {
    const created = await createResearchAgentRun(
      researchScope.value.scopeId,
      `研究报告：${question.slice(0, 120)}`,
      'Use the Host-confirmed ResearchScope and exchange only structured evidence IDs.',
      question,
    )
    const detail = await runReviewCaseUntilReview(created.case.caseId)
    researchAgentDetail.value = detail
    if (detail.case.status === 'failed') {
      throw new Error(detail.case.failure?.reason || '研究报告执行失败，请稍后重试。')
    }
    if (detail.case.status !== 'waiting_human_review') {
      throw new Error(`报告流程尚未完成，当前状态：${readableStatus(detail.case.status)}。请重新生成或查看失败原因。`)
    }
    try {
      const allEvidence = await listResearchEvidence(
        researchScope.value.scopeId,
        researchScope.value.scopeVersion,
      )
      const cited = new Set(detail.researchAnswer?.evidenceIds ?? [])
      reportEvidence.value = allEvidence.filter((item) => cited.has(item.evidenceId))
    } catch {
      // A report remains readable when source-label hydration is temporarily
      // unavailable; numbered citations still preserve their stable order.
      reportEvidence.value = []
    }
    researchMessage.value = '研究报告草稿已生成，请核对引用。'
  } catch (error) {
    researchError.value = error instanceof Error ? error.message : '研究报告生成失败。'
    researchMessage.value = ''
  } finally {
    reportGenerating.value = false
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
  window.addEventListener('keydown', handleEvidenceKeydown)
})
onBeforeUnmount(() => {
  stopPolling()
  window.removeEventListener('keydown', handleEvidenceKeydown)
})
</script>

<template>
  <main class="page-shell">
    <header class="hero app-header">
      <div class="app-brand">
        <div class="brand-mark" aria-hidden="true">TF</div>
        <div>
          <p class="eyebrow">TASKFORGE</p>
          <h1>论文研究助手</h1>
        </div>
      </div>
      <div class="app-header-actions">
        <span class="service-status"><i aria-hidden="true" />学术检索服务已连接</span>
        <button
          v-if="literatureResult || researchScope"
          class="new-research-button"
          type="button"
          :disabled="researchLoading"
          @click="startNewResearch"
        >新建研究</button>
      </div>
    </header>

    <template v-if="activeMode === 'research'">
      <div class="research-status-stack">
        <div v-if="researchError" class="notice notice-error" role="alert">
          <strong>这次研究没有完成</strong><span>{{ researchError }}</span>
        </div>
        <div v-if="researchMessage" class="notice research-notice" role="status">
          <strong>当前进度</strong><span>{{ researchMessage }}</span>
        </div>
      </div>

      <section
        class="research-home"
        :class="{ compact: Boolean(literatureResult || researchScope) }"
        data-stage="consensus-home"
      >
        <div v-if="!literatureResult && !researchScope" class="research-home-copy">
          <span class="home-kicker">基于真实论文的研究工作台</span>
          <h2>从论文出发，找到可信答案</h2>
          <p>发现中英文研究，核对全文证据，再生成带原文引用的中文报告。</p>
        </div>
        <div v-else class="research-home-context">
          <span>当前研究</span>
          <strong>{{ researchIntent || literatureQuery }}</strong>
        </div>

        <div class="query-composer">
          <textarea
            id="research-query"
            v-model="literatureQuery"
            rows="2"
            maxlength="4000"
            :disabled="researchLoading"
            aria-label="研究问题"
            placeholder="输入一个研究问题，例如：RAG 的最新技术方向与评测方法有哪些？"
            @keydown.ctrl.enter.prevent="handleLiteratureSearch"
            @keydown.meta.enter.prevent="handleLiteratureSearch"
          />
          <div class="query-composer-footer">
            <div class="language-choice" role="group" aria-label="论文语言偏好">
              <button
                v-for="choice in [
                  { value: 'balanced', label: '综合' },
                  { value: 'chinese_first', label: '中文优先' },
                  { value: 'english_first', label: '英文优先' },
                ]"
                :key="choice.value"
                type="button"
                :class="{ active: languagePreference === choice.value }"
                :aria-pressed="languagePreference === choice.value"
                @click="languagePreference = choice.value as LiteratureLanguagePreference"
              >{{ choice.label }}</button>
            </div>
            <span class="query-source-note">四个开放学术数据源</span>
            <button
              class="query-submit"
              data-action="search-literature"
              :disabled="researchLoading || !literatureQuery.trim()"
              @click="handleLiteratureSearch"
            >
              <span>{{ researchLoading ? '正在查找论文…' : literatureResult ? '重新检索' : '查找论文' }}</span>
              <i aria-hidden="true">→</i>
            </button>
          </div>
        </div>

        <div v-if="!literatureResult && !researchScope" class="example-queries" aria-label="示例研究问题">
          <span>试试这些问题</span>
          <button v-for="query in exampleResearchQueries" :key="query" type="button" @click="useExampleQuery(query)">
            {{ query }}
          </button>
        </div>
      </section>

      <nav v-if="literatureResult || researchScope" class="research-journey" aria-label="论文研究流程">
        <ol>
          <li
            v-for="step in researchSteps"
            :key="step.number"
            :data-state="step.number < researchStep ? 'done' : step.number === researchStep ? 'active' : 'pending'"
          >
            <span>{{ step.number < researchStep ? '✓' : step.number }}</span>
            <strong>{{ step.title }}</strong>
          </li>
        </ol>
        <p>{{ researchStepGuide }}</p>
      </nav>

      <details v-if="!researchScope" class="research-options" data-panel="advanced-filters">
        <summary>
          <div><strong>筛选条件与 PDF 上传</strong><span>年份、关键词、返回数量和数据源状态</span></div>
          <i aria-hidden="true">⌄</i>
        </summary>
        <div class="research-options-content">
          <section class="direct-upload-card">
            <div><strong>已有论文 PDF？</strong><p>使用当前研究问题，直接上传并建立全文索引。</p></div>
            <label class="field">
              <span>论文标题（可选）</span>
              <input v-model="directUploadTitle" maxlength="2000" :disabled="researchLoading" placeholder="默认使用文件名" />
            </label>
            <label class="secondary-button direct-upload-button">
              上传 PDF
              <input
                type="file"
                accept="application/pdf,.pdf"
                :disabled="researchLoading || (!literatureQuery.trim() && !researchIntent.trim())"
                @change="handleDirectResearchUpload"
              />
            </label>
          </section>
          <div class="advanced-filter-grid">
            <label class="field advanced-question"><span>补充研究问题（可选，逗号分隔）</span><textarea v-model="researchQuestionsText" rows="2" maxlength="2000" :disabled="researchLoading" placeholder="例如：有哪些代表性方法？如何评测？" /></label>
            <label class="field"><span>起始年份</span><input v-model.number="yearFrom" type="number" min="1000" max="3000" /></label>
            <label class="field"><span>结束年份</span><input v-model.number="yearTo" type="number" min="1000" max="3000" /></label>
            <label class="field"><span>返回数量</span><select v-model.number="literatureResultLimit"><option :value="30">30 篇</option><option :value="50">50 篇</option><option :value="100">100 篇</option></select></label>
            <label class="field"><span>必须包含</span><input v-model="requiredTermsText" placeholder="多个词用逗号分隔" /></label>
            <label class="field"><span>排除词</span><input v-model="excludedTermsText" placeholder="多个词用逗号分隔" /></label>
          </div>
          <details v-if="literatureResult" class="provider-health provider-health-details">
            <summary>数据源运行状态</summary>
            <article v-for="provider in literatureResult.providers" :key="provider.provider" :data-failed="Boolean(provider.failure)">
              <div><strong>{{ providerLabel(provider.provider) }}</strong><span>{{ provider.failure ? '部分不可用' : '正常' }}</span></div>
              <small>{{ provider.resultCount }} 条结果 · {{ Math.round(provider.elapsedMs) }} 毫秒</small>
              <small v-if="provider.failure">{{ provider.failure }}</small>
            </article>
          </details>
        </div>
      </details>

      <section v-if="literatureResult && !researchScope" class="research-workspace" data-stage="paper-selection">

        <section class="research-results panel" data-list="papers">
          <div class="section-heading research-results-heading">
            <div>
              <span class="section-kicker">论文发现</span>
              <h2>找到 {{ literatureResult?.papers.length ?? 0 }} 篇相关论文</h2>
              <p>按匹配度排序，其中中文 {{ chinesePaperCount }} 篇；来源信息有限的条目会明确标记。</p>
            </div>
            <button class="secondary-button" :disabled="researchLoading || !selectedPaperIds.length" @click="handleCitationExpansion">沿引用继续查找</button>
          </div>

          <div v-if="!literatureResult" class="research-empty">
            <strong>输入研究需求开始</strong>
            <p>系统会从多个学术数据源查找相关论文，并保留可以核对的来源信息。</p>
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
              :data-paper-id="paper.paperId"
            >
              <div class="paper-copy">
                <div class="paper-card-topline">
                  <div class="paper-badges">
                    <span class="paper-trust" :data-tone="paperTrustTone(paper)">{{ paperTrustLabel(paper) }}</span>
                    <span v-if="publicationTypeLabel(paper.publicationType)">{{ publicationTypeLabel(paper.publicationType) }}</span>
                    <span v-if="paperLanguageLabel(paper)">{{ paperLanguageLabel(paper) }}</span>
                    <span v-if="paper.year">{{ paper.year }}</span>
                  </div>
                  <span class="paper-score"><strong>{{ Math.round(paper.relevanceScore * 100) }}%</strong> 匹配</span>
                </div>
                <h3>{{ paper.title }}</h3>
                <p class="paper-authors">{{ paper.authors.slice(0, 5).join(', ') || '作者信息缺失' }}<template v-if="paper.venue"> · {{ paper.venue }}</template><template v-if="paper.publisher && paper.publisher !== paper.venue"> · {{ paper.publisher }}</template></p>
                <p class="paper-abstract">{{ paper.shortDescription || '暂无可验证的简短介绍。' }}</p>
                <div class="paper-card-footer">
                  <div class="paper-links">
                    <a v-for="source in paper.sourceUrls.slice(0, 2)" :key="source" :href="source" target="_blank" rel="noreferrer">查看来源 ↗</a>
                    <span>{{ fullTextLabel(paper.fullTextStatus) }}</span>
                    <span v-if="paper.citationCount !== undefined">被引 {{ paper.citationCount }}</span>
                  </div>
                  <button
                    class="paper-select"
                    data-action="toggle-paper"
                    :disabled="Boolean(researchScope)"
                    :aria-pressed="selectedPaperIds.includes(paper.paperId)"
                    @click="togglePaper(paper.paperId)"
                  >
                    {{ selectedPaperIds.includes(paper.paperId) ? '✓ 已加入' : '+ 加入论文库' }}
                  </button>
                </div>
              </div>
            </article>
          </div>
        </section>

        <aside class="scope-panel panel" data-list="selected-papers">
          <div class="section-heading">
            <div><span class="section-kicker">我的论文库</span><h2>{{ selectedPaperIds.length }} 篇论文</h2></div>
            <button v-if="selectedPaperIds.length" class="clear-selection" type="button" @click="clearSelectedPapers">清空</button>
          </div>
          <template v-if="!researchScope">
            <p class="scope-summary">先加入与你的问题真正相关、且来源能够核对的论文。</p>
            <p v-if="!selectedPapers.length" class="selected-empty">还没有选择论文。点击结果中的“加入论文库”即可开始。</p>
            <ul v-else class="selected-paper-list">
              <li v-for="paper in selectedPapers" :key="paper.paperId">
                <span><strong>{{ paper.title }}</strong><small>{{ paper.year || '年份未知' }} · {{ paperTrustLabel(paper) }}</small></span>
                <button type="button" :aria-label="`移除 ${paper.title}`" @click="togglePaper(paper.paperId)">移除</button>
              </li>
            </ul>
            <label class="field scope-goal"><span>研究目标</span><textarea v-model="researchIntent" rows="3" maxlength="4000" placeholder="系统将围绕这个问题准备全文证据" /></label>
            <details class="scope-extra-setting">
              <summary>更多设置</summary>
              <label class="scope-checkbox"><input v-model="allowExpansion" type="checkbox" /><span>允许后续补充相关论文（执行前仍需确认）</span></label>
            </details>
            <button class="primary-button scope-confirm" data-action="save-scope" :disabled="researchLoading || !selectedPaperIds.length || !researchIntent.trim()" @click="handleConfirmScope">
              {{ researchLoading ? '正在准备…' : '确认论文并获取全文' }}
            </button>
            <small class="scope-legal-note">仅自动获取合法开放的 PDF；不会绕过登录或付费限制。</small>
          </template>
        </aside>
      </section>

      <section v-if="researchScope" class="research-studio">
        <header class="studio-header">
          <div class="studio-summary">
            <span class="section-kicker">当前研究空间</span>
            <strong>{{ researchScope.selectedPaperIds.length }} 篇论文</strong>
            <small>{{ indexedPaperCount }} 篇已建立全文索引</small>
          </div>
          <nav class="studio-tabs" aria-label="研究空间">
            <button type="button" :data-active="researchWorkspaceView === 'sources'" @click="showResearchView('sources')">论文库</button>
            <button type="button" :data-active="researchWorkspaceView === 'ask'" :disabled="researchScope.status !== 'ready'" @click="showResearchView('ask')">证据问答</button>
            <button type="button" :data-active="researchWorkspaceView === 'report'" :disabled="researchScope.status !== 'ready'" @click="showResearchView('report')">研究报告</button>
          </nav>
          <button
            class="studio-report-shortcut"
            type="button"
            :disabled="researchScope.status !== 'ready'"
            @click="showResearchView('report')"
          >生成研究报告 <span aria-hidden="true">→</span></button>
        </header>

        <section v-if="researchWorkspaceView === 'sources'" class="studio-panel sources-workspace" data-stage="sources">
          <div class="sources-overview">
            <span class="sources-progress-icon" :data-ready="researchScope.status === 'ready'">{{ researchScope.status === 'ready' ? '✓' : indexedPaperCount }}</span>
            <div>
              <span class="section-kicker">全文准备</span>
              <h2>{{ researchScope.status === 'ready' ? '论文全文已经就绪' : '正在准备论文全文' }}</h2>
              <p>{{ researchScope.status === 'ready' ? '现在可以检索原文证据，或生成带引用的研究报告。' : '开放论文会自动下载；受限论文需要你从来源页面下载后上传。' }}</p>
            </div>
          </div>
          <div class="source-goal"><span>研究目标</span><p>{{ researchScope.userIntent }}</p></div>
          <div class="source-library-list">
            <article v-for="paper in selectedPapers" :key="paper.paperId">
              <div class="source-paper-status" :data-status="ingestionForPaper(paper.paperId)?.status || 'queued'">
                <i aria-hidden="true" />{{ ingestionStatusLabel(ingestionForPaper(paper.paperId)?.status || 'queued') }}
              </div>
              <h3>{{ paper.title }}</h3>
              <p>{{ paper.authors.slice(0, 4).join(', ') || '作者信息缺失' }}<template v-if="paper.year"> · {{ paper.year }}</template></p>
              <small v-if="ingestionForPaper(paper.paperId)?.error">{{ ingestionForPaper(paper.paperId)?.error }}</small>
              <div v-if="ingestionForPaper(paper.paperId)?.status !== 'indexed'" class="source-paper-actions">
                <a v-for="source in paper.sourceUrls.slice(0, 2)" :key="source" :href="source" target="_blank" rel="noreferrer">打开论文来源 ↗</a>
                <label class="source-upload-button">
                  {{ uploadedPaperIds.includes(paper.paperId) ? 'PDF 已上传' : '上传对应 PDF' }}
                  <input type="file" accept="application/pdf,.pdf" :disabled="researchLoading || uploadedPaperIds.includes(paper.paperId)" @change="handlePaperUpload(paper.paperId, $event)" />
                </label>
              </div>
            </article>
          </div>
          <button
            v-if="researchScope.status !== 'ready'"
            class="primary-button source-retry"
            :disabled="researchLoading || !canRetryScopeIngestion"
            @click="handleScopeIngestion"
          >{{ researchLoading ? '正在解析…' : '解析已上传的 PDF 并刷新状态' }}</button>
        </section>

        <div v-else-if="researchWorkspaceView === 'ask'" class="notebook-layout" data-stage="evidence">
          <aside class="notebook-sources">
            <header><span class="section-kicker">论文来源</span><strong>{{ selectedPapers.length }} 篇</strong></header>
            <ul>
              <li v-for="paper in selectedPapers" :key="paper.paperId">
                <i aria-hidden="true">{{ ingestionForPaper(paper.paperId)?.status === 'indexed' ? '✓' : '•' }}</i>
                <span><strong>{{ paper.title }}</strong><small>{{ paper.year || '年份未知' }} · {{ paperLanguageLabel(paper) || '语言未知' }}</small></span>
              </li>
            </ul>
          </aside>
          <section class="ask-workspace">
            <header class="workspace-heading">
              <div><span class="section-kicker">基于已选论文</span><h2>查找可以核对的原文证据</h2><p>回答范围严格限制在当前论文库，并保留页码与原文位置。</p></div>
            </header>
            <div class="ask-composer">
              <textarea id="evidence-query" v-model="evidenceQuery" rows="3" maxlength="4000" placeholder="例如：这些论文采用了哪些方法提高检索准确率？" />
              <div class="ask-composer-footer">
                <label><span>问题类型</span>
                  <select id="evidence-intent" v-model="evidenceIntent">
                    <option value="general_fact">一般事实</option><option value="method_definition">方法定义</option>
                    <option value="experimental_setup">实验设置</option><option value="numeric_table">数值与表格</option>
                    <option value="cross_paper_comparison">跨论文比较</option><option value="claim_verification">论断核验</option>
                    <option value="related_work">相关工作</option>
                  </select>
                </label>
                <button class="primary-button" :disabled="researchLoading || !evidenceQuery.trim()" @click="handleEvidenceSearch">
                  {{ researchLoading ? '正在检索…' : '查找原文证据' }}
                </button>
              </div>
            </div>

            <div v-if="scopeEvidence" class="evidence-results">
              <div class="evidence-summary" :data-sufficient="scopeEvidence.confidence.sufficient">
                <div><strong>{{ scopeEvidence.confidence.sufficient ? '证据充分' : '建议继续核对' }}</strong><span>{{ scopeEvidence.confidence.citationReadyCount }} 条可引用片段</span></div>
                <p>已有 {{ formatPercent(scopeEvidence.confidence.scopePaperCoverage) }} 的论文至少命中一个片段；这不代表全文内容已被完整覆盖。</p>
              </div>
              <div class="research-evidence-list">
                <article v-for="(item, evidenceIndex) in visibleEvidence" :key="item.evidenceId" :data-evidence-id="item.evidenceId">
                  <header><span>证据 {{ evidenceIndex + 1 }} · {{ item.section || '正文' }}</span><strong>{{ Math.round(item.score * 100) }}% 匹配</strong></header>
                  <h3>{{ item.title || paperTitle(item.paperId) }}</h3>
                  <p class="evidence-snippet">
                    <template v-for="(segment, index) in highlightEvidenceText(item.snippet)" :key="`${item.evidenceId}-${index}`">
                      <mark v-if="segment.highlighted">{{ segment.text }}</mark>
                      <template v-else>{{ segment.text }}</template>
                    </template>
                  </p>
                  <footer>
                    <span>{{ item.page ? `第 ${item.page} 页` : '页码未知' }}</span>
                    <button class="evidence-view-button" type="button" @click="openEvidenceDrawer(item)">核对完整原文 →</button>
                  </footer>
                </article>
                <button v-if="hasMoreEvidence" class="evidence-list-toggle" type="button" :aria-expanded="showAllEvidence" @click="showAllEvidence = !showAllEvidence">
                  {{ showAllEvidence ? '收起其余证据' : `再看 ${sortedEvidence.length - 5} 条证据` }}
                </button>
              </div>
            </div>
            <div v-else class="ask-empty-state"><span aria-hidden="true">⌕</span><strong>先提出一个具体问题</strong><p>系统会返回短小、可追溯的原文片段，不会用目录或参考文献凑数量。</p></div>
          </section>
        </div>

        <section v-else class="studio-panel report-workspace" data-stage="report">
          <header class="report-workspace-heading">
            <div><span class="section-kicker">中文研究报告</span><h2>把论文证据整理成可核对的报告</h2><p>{{ reportAvailabilityReason }}</p></div>
          </header>
          <div class="report-composer">
            <label for="report-question">这份报告需要回答什么？</label>
            <textarea id="report-question" v-model="reportQuestion" rows="3" maxlength="4000" placeholder="例如：从已选论文看，RAG 的最新技术方向有哪些？" />
            <div><span>仅使用当前 {{ researchScope.selectedPaperIds.length }} 篇论文，默认使用中文撰写。</span>
              <button
                class="primary-button generate-report-button"
                data-action="generate-report"
                :disabled="researchLoading || researchScope.status !== 'ready' || !reportQuestion.trim()"
                @click="handleResearchAgents"
              >{{ reportGenerating ? '正在生成报告…' : researchAgentDetail?.case.status === 'failed' ? '重新生成报告' : '生成研究报告' }}</button>
            </div>
          </div>
          <ol v-if="reportGenerating" class="report-role-progress" aria-label="报告生成进度">
            <li v-for="role in ['理解研究问题', '筛选原文证据', '撰写中文草稿', '核对论断与引用']" :key="role"><i />{{ role }}</li>
          </ol>
          <article v-if="researchAgentDetail" class="research-report">
            <header><span>研究报告</span><strong>{{ researchAgentDetail.case.status === 'failed' ? '生成失败' : researchAgentDetail.case.status === 'waiting_human_review' ? '引用待核对' : '处理中' }}</strong></header>
            <h2>{{ researchAgentDetail.case.title }}</h2>
            <p v-if="researchAgentDetail.case.failure" class="research-report-summary">{{ researchAgentDetail.case.failure.reason }}</p>
            <template v-else-if="researchAgentDetail.researchAnswer">
              <p v-if="researchAgentDetail.researchAnswer.scopeNote" class="research-report-scope"><strong>证据范围提示</strong>{{ researchAgentDetail.researchAnswer.scopeNote }}</p>
              <p v-if="researchAgentDetail.researchAnswer.directAnswer && researchAgentDetail.researchAnswer.directAnswer !== researchAgentDetail.researchAnswer.scopeNote" class="research-report-direct"><strong>基于已选论文的结论</strong>{{ researchAgentDetail.researchAnswer.directAnswer }}</p>
              <div class="research-report-body" data-content="report-body">
                <p v-for="(paragraph, index) in reportParagraphs(researchAgentDetail.researchAnswer.answer)" :key="`${researchAgentDetail.case.caseId}-report-${index}`">{{ paragraph }}</p>
              </div>
            </template>
            <p v-else class="research-report-summary">{{ writerSummary(researchAgentDetail) || '研究报告正在整理。' }}</p>
            <div v-if="reportSources.length" class="research-report-sources">
              <span>引用原文</span>
              <button v-for="source in reportSources" :key="source.evidenceId" type="button" :data-source-id="source.evidenceId" :disabled="!source.card" @click="source.card && openEvidenceDrawer(source.card)">{{ reportSourceLabel(source) }}</button>
            </div>
            <details v-if="researchAgentDetail.case.recommendation" class="research-report-review" data-panel="report-review">
              <summary>查看报告核验详情</summary>
              <p>{{ researchAgentDetail.case.recommendation.summary }}</p>
              <p v-if="researchAgentDetail.case.recommendation.rationale">{{ researchAgentDetail.case.recommendation.rationale }}</p>
            </details>
          </article>
          <div v-else class="report-empty-state"><span aria-hidden="true">✦</span><strong>一键生成带原文引用的中文草稿</strong><p>生成后仍会明确提示证据范围，方便逐条核对引用。</p></div>
        </section>
      </section>
    </template>

    <Teleport to="body">
      <div
        v-if="activeEvidence"
        class="evidence-drawer-backdrop"
        role="presentation"
        @click.self="closeEvidenceDrawer"
      >
        <aside
          class="evidence-drawer"
          data-panel="evidence-drawer"
          role="dialog"
          aria-modal="true"
          aria-labelledby="evidence-drawer-title"
        >
          <header class="evidence-drawer-header">
            <div>
              <p>完整证据片段</p>
              <h2 id="evidence-drawer-title">{{ activeEvidence.title || paperTitle(activeEvidence.paperId) }}</h2>
            </div>
            <button class="evidence-drawer-close" type="button" aria-label="关闭完整证据" @click="closeEvidenceDrawer">关闭 ×</button>
          </header>
          <dl class="evidence-drawer-meta">
            <div><dt>页码</dt><dd>{{ activeEvidence.page ? `第 ${activeEvidence.page} 页` : '页码未知' }}</dd></div>
            <div><dt>章节</dt><dd>{{ activeEvidence.section || '正文' }}</dd></div>
          </dl>
          <p class="evidence-drawer-text">{{ activeEvidence.snippet }}</p>
          <details class="evidence-technical-details">
            <summary>技术信息</summary>
            <span>Evidence ID</span>
            <code>{{ activeEvidence.evidenceId }}</code>
          </details>
        </aside>
      </div>
    </Teleport>

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
