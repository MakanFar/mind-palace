import { App, Plugin, PluginSettingTab, Setting } from "obsidian";

import { DEFAULT_SETTINGS, MindPalaceView, VIEW_TYPE, type Settings } from "./view";

export default class MindPalacePlugin extends Plugin {
  settings: Settings = { ...DEFAULT_SETTINGS };

  async onload(): Promise<void> {
    this.settings = { ...DEFAULT_SETTINGS, ...((await this.loadData()) ?? {}) };
    this.registerView(VIEW_TYPE, (leaf) => new MindPalaceView(leaf, () => this.settings));
    this.addRibbonIcon("network", "Open Mind Palace", () => void this.activateView());
    this.addCommand({
      id: "open-mind-palace",
      name: "Open Mind Palace",
      callback: () => void this.activateView(),
    });
    this.addSettingTab(new MindPalaceSettingTab(this.app, this));
  }

  async activateView(): Promise<void> {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    if (existing.length) {
      await this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    const leaf = this.app.workspace.getLeaf("tab");
    await leaf.setViewState({ type: VIEW_TYPE, active: true });
    await this.app.workspace.revealLeaf(leaf);
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }
}

class MindPalaceSettingTab extends PluginSettingTab {
  constructor(
    app: App,
    private readonly plugin: MindPalacePlugin,
  ) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl)
      .setName("Graph file")
      .setDesc("Vault-relative path of the graph.json the Mind Palace server writes.")
      .addText((text) =>
        text.setValue(this.plugin.settings.graphPath).onChange(async (value) => {
          this.plugin.settings.graphPath = value.trim() || DEFAULT_SETTINGS.graphPath;
          await this.plugin.saveSettings();
        }),
      );
    new Setting(containerEl)
      .setName("Retirements file")
      .setDesc("Vault-relative path of retirements.jsonl; retiring an entity appends here.")
      .addText((text) =>
        text.setValue(this.plugin.settings.retirementsPath).onChange(async (value) => {
          this.plugin.settings.retirementsPath = value.trim() || DEFAULT_SETTINGS.retirementsPath;
          await this.plugin.saveSettings();
        }),
      );
    new Setting(containerEl)
      .setName("Decisions file")
      .setDesc("Vault-relative path of decisions.jsonl; confirm and dismiss append here.")
      .addText((text) =>
        text.setValue(this.plugin.settings.decisionsPath).onChange(async (value) => {
          this.plugin.settings.decisionsPath = value.trim() || DEFAULT_SETTINGS.decisionsPath;
          await this.plugin.saveSettings();
        }),
      );
  }
}
