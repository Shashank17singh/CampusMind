export type Role = "student" | "faculty" | "admin";

export interface User {
  id: string;
  email: string;
  full_name: string;
  roll_number: string | null;
  role: Role;
  department: string | null;
  semester: number | null;
  section: string | null;
  timezone: string;
  is_active: boolean;
  email_verified_at: string | null;
  created_at: string;
}

export interface Page<T> {
  items: T[];
  total: number;
}

export interface Citation {
  /** Matches the [n] marker in the answer text. */
  number: number;
  document_id: string;
  title: string;
  page: number | null;
  snippet: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  last_message_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations: Citation[];
  intent: string | null;
  confidence: "high" | "medium" | "low" | null;
  created_at: string;
}

export interface SearchHit {
  message_id: string;
  conversation_id: string;
  snippet: string;
  created_at: string;
}

export type FeedbackReason = "wrong" | "outdated" | "incomplete" | "not_relevant";

export interface PlanItem {
  id: string;
  date: string;
  start_time: string;
  end_time: string;
  subject: string;
  topic: string;
  type: "learn" | "revise" | "practice";
  status: "pending" | "done" | "missed";
}

export interface PlanSummary {
  id: string;
  title: string;
  start_date: string;
  end_date: string;
  is_active: boolean;
  created_at: string;
}

export interface StudyPlan extends PlanSummary {
  items: PlanItem[];
  warnings: string[];
}

export interface Reminder {
  id: string;
  title: string;
  subject: string | null;
  due_at: string;
  remind_before: number[];
  recurrence: string | null;
  status: "pending" | "done" | "cancelled";
  snoozed_until: string | null;
  task_id: string | null;
}

export type ReminderView = "upcoming" | "overdue" | "done";

export interface ParseResult {
  reminder: Reminder | null;
  clarification: string | null;
}

export type DocType = "syllabus" | "circular" | "notice" | "regulation" | "notes" | "contact_list";

export interface DocumentRecord {
  id: string;
  title: string;
  filename: string;
  size_bytes: number;
  doc_type: DocType | null;
  department: string | null;
  semester: number | null;
  status: "uploaded" | "extracting" | "chunking" | "embedding" | "ready" | "failed";
  progress: number;
  error_message: string | null;
  page_count: number | null;
  uploader_id: string | null;
  superseded_by_id: string | null;
  uploaded_at: string;
}

export interface UploadResult extends DocumentRecord {
  duplicate_of: string | null;
}

export type SummaryMode = "brief" | "detailed" | "exam_notes" | "flashcards";

export interface SummaryItem {
  text: string;
  answer: string | null;
  pages: number[];
}

export interface SummaryGroup {
  heading: string;
  items: SummaryItem[];
}

export interface Summary {
  id: string;
  document_id: string | null;
  source_name: string;
  mode: SummaryMode;
  status: "pending" | "ready" | "failed";
  error_message: string | null;
  content: { groups: SummaryGroup[]; verified?: boolean } | null;
  dropped_count: number;
  created_at: string;
}

export interface Assignment {
  id: string;
  title: string;
  subject: string | null;
  department: string;
  semester: number;
  section: string | null;
  due_at: string;
  reminders_created: number | null;
}

export interface Overview {
  daily: { day: string; active_users: number; questions: number }[];
  avg_latency_ms: number | null;
  thumbs_up: number;
  thumbs_down: number;
  feedback_ratio: number | null;
}

export interface QuestionCluster {
  question?: string;
  topic?: string;
  count: number;
  examples: string[];
}
