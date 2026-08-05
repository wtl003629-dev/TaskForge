export interface SkillPack {
  id: string
  name: string
  description: string
  tools: string[]
}

export interface AgentProfile {
  id: string
  name: string
  description: string
  skillPacks: SkillPack[]
}

export type RunStatus =
  | 'queued'
  | 'running'
  | 'waiting_approval'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | string

export interface ToolCall {
  id: string
  name: string
  status: string
  arguments?: Record<string, unknown>
  result?: string
  requiresApproval?: boolean
}

export interface Evidence {
  id: string
  title: string
  kind: string
  source?: string
  summary: string
}

export interface RunStep {
  id: string
  title: string
  status: string
  summary?: string
  toolCalls: ToolCall[]
  evidence: Evidence[]
}

export interface PendingApproval {
  id: string
  toolName: string
  reason: string
  risk: string
  arguments?: Record<string, unknown>
}

export interface RunRecord {
  id: string
  task: string
  agentId: string
  skillPackId: string
  status: RunStatus
  createdAt: string
  updatedAt: string
  steps: RunStep[]
  evidence: Evidence[]
  pendingApproval?: PendingApproval
  summary?: string
}

export type ExecutionMode = 'inline' | 'queued'

export interface CreateRunInput {
  task: string
  agentId: string
  skillPackId: string
  executionMode: ExecutionMode
}

export type JobStatus = 'queued' | 'leased' | 'completed' | 'dead_letter' | string

export interface OperationJob {
  runId: string
  status: JobStatus
  attempt: number
  maxAttempts: number
  availableAt: string
  leaseExpiresAt?: string
  resultStatus?: string
  lastError?: string
  updatedAt: string
}

export interface AuditUsage {
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  costUsd?: number
}

export interface AuditEvent {
  eventId: string
  action: string
  outcome: string
  occurredAt: string
  durationMs?: number
  tool?: string
  provider?: string
  usage?: AuditUsage
  safetyViolation: boolean
}

export interface MetricsSnapshot {
  runId?: string
  runCount: number
  runSuccessCount: number
  runSuccessRate?: number
  toolCount: number
  toolSuccessCount: number
  toolSuccessRate?: number
  durationP50Ms?: number
  durationP95Ms?: number
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  costUsd?: number
  safetyViolationCount: number
}

export interface McpServerSummary {
  namespace: string
  enabled: boolean
  profileIds: string[]
  configuredTools: string[]
  mountedTools: string[]
}

export interface ApiResult<T> {
  data: T
  mock: boolean
  warning?: string
}

export type ReviewCaseKind = 'enterprise_change' | 'enterprise_admission'

export type ReviewCaseStatus =
  | 'draft'
  | 'submitted'
  | 'running'
  | 'waiting_human_review'
  | 'approved'
  | 'rejected'
  | 'failed'
  | string

export interface ReviewEvidenceRef {
  evidenceId: string
  sourceType: 'document' | 'artifact' | 'tool_receipt' | 'url' | 'case' | string
  locator: string
  excerpt: string
  title?: string
  version?: string
  checksumSha256?: string
  pageNumber?: number
}

export interface ReviewSubmission {
  requestSummary: string
  businessJustification: string
  attributes: Record<string, unknown>
  evidenceRefs: ReviewEvidenceRef[]
}

export interface ReviewRecommendation {
  recommendationId: string
  modelRunId: string
  modelId: string
  outcome: 'approve' | 'reject' | 'escalate' | string
  summary: string
  rationale: string
  confidence: number
  evidenceRefs: ReviewEvidenceRef[]
  authority: 'model_untrusted'
  producedAt: string
}

export interface ReviewHumanDecision {
  outcome: 'approved' | 'rejected'
  actor: {
    displayName?: string
    authority: 'human'
  }
  rationale: string
  evidenceRefIds: string[]
  decidedAt: string
}

export interface ReviewCase {
  caseId: string
  conversationId: string
  kind: ReviewCaseKind
  title: string
  submission: ReviewSubmission
  status: ReviewCaseStatus
  recommendation?: ReviewRecommendation
  humanDecision?: ReviewHumanDecision
  failure?: {
    reason: string
    failedAt: string
  }
  revision: number
  createdAt: string
  updatedAt: string
  submittedAt?: string
  startedAt?: string
  reviewRequestedAt?: string
  resolvedAt?: string
}

export interface ReviewExecutionDisclosure {
  provider: string
  mode: 'offline-deterministic-demo' | 'configured-provider' | 'injected-test-provider'
  providerConfigured: boolean
  contractTestedMock: boolean
  liveSmokeVerified: boolean
  businessE2eVerified: boolean
  recommendationAuthority: 'model_untrusted'
  finalDecisionAuthority: 'human'
}

export interface ReviewPlanSlot {
  slotId: string
  roleId: string
  agentProfileId: string
  dependsOn: string[]
  order: number
  required: boolean
  maxAttempts: number
}

export interface ReviewPlan {
  planId: string
  status: string
  version: number
  slots: ReviewPlanSlot[]
  createdAt: string
  updatedAt: string
}

export interface ReviewRoleRun {
  roleRunId: string
  slotId: string
  roleId: string
  agentProfileId: string
  attempt: number
  status: string
  version: number
  runtimeStatus?: string
  summary?: string
  summaryAuthority?: string
  citations: string[]
  retrievedEvidenceRefs: string[]
  runtimeMetrics?: {
    stepCount: number
    modelTurnCount: number
    toolCallCount: number
    toolResultCount: number
    toolSuccessCount: number
    toolFailureCount: number
    safetyViolationCount: number
    elapsedMs: number
    usage?: AuditUsage
  }
  pendingApprovalCallId?: string
  roleResult?: Record<string, unknown>
  error?: string
  createdAt: string
  updatedAt: string
}

export interface ReviewFact {
  factId: string
  factKey: string
  value: unknown
  status: 'proposed' | 'verified' | string
  authority: 'model' | 'tool' | 'user' | 'system' | string
  version: number
  sourceRoleRunId?: string
  verifierRef?: string
  createdAt: string
}

export interface ReviewCaseAuditEvent {
  eventId: string
  eventType: string
  revision: number
  fromStatus?: string
  toStatus: string
  actorAuthority: string
  details: Record<string, unknown>
  createdAt: string
}

export interface ReviewHandoff {
  handoffId: string
  fromRoleRunId: string
  toSlotId: string
  summary: string
  sharedFactIds: string[]
  createdAt: string
}

export interface ReviewCaseDetail {
  case: ReviewCase
  plan?: ReviewPlan
  roleRuns: ReviewRoleRun[]
  sharedFacts: ReviewFact[]
  handoffs: ReviewHandoff[]
  auditEvents: ReviewCaseAuditEvent[]
  execution: ReviewExecutionDisclosure
}

export interface ReviewCaseList {
  items: ReviewCase[]
  execution: ReviewExecutionDisclosure
}

export interface CreateReviewCaseInput {
  kind: ReviewCaseKind
  title: string
  requestSummary: string
  businessJustification: string
  evidenceId: string
  evidenceLocator: string
  evidenceExcerpt: string
}

export interface ReviewDecisionInput {
  expectedRevision: number
  outcome: 'approved' | 'rejected'
  rationale: string
  evidenceRefIds: string[]
}
