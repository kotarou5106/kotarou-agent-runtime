// Type declarations for globals injected by the React dashboard bundle.
// Dashboard plugin interfaces are defined in dashboard/frontend/src/types.ts.

import type { DashboardColumn, DashboardGlobal, FetchPageOpts, FetchPageResult, PluginBatchAction, PluginConfig, PluginDispatch, SortOrder } from "../dashboard/frontend/src/types";

export type Column = DashboardColumn;
export type { DashboardColumn, DashboardGlobal, FetchPageOpts, FetchPageResult, PluginBatchAction, PluginConfig, PluginDispatch, SortOrder };

declare global {
  interface Window {
    KotarouDashboard: DashboardGlobal;
  }

  function api<T = unknown>(url: string, options?: RequestInit): Promise<T>;
  function escapeHtml(s: string): string;
  function encodePath(s: string): string;
  function renderMarkdown(s: string): string;
  function makeJsonViewer(data: unknown): HTMLElement;
  function jvPlaceholder(data: unknown): string;
  function attachJsonViewers(container: ParentNode): void;

  // Plugin protocol types — available in all panel.ts files via /// <reference path>
  type PluginDispatch = import("../dashboard/frontend/src/types").PluginDispatch;
  type PluginBatchAction = import("../dashboard/frontend/src/types").PluginBatchAction;
  type FetchPageOpts = import("../dashboard/frontend/src/types").FetchPageOpts;
  type FetchPageResult = import("../dashboard/frontend/src/types").FetchPageResult;
  type SortOrder = import("../dashboard/frontend/src/types").SortOrder;
}
