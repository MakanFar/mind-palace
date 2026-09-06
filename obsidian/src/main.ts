import { Plugin } from "obsidian";

export default class MindPalacePlugin extends Plugin {
  async onload(): Promise<void> {
    console.log("mind-palace: loaded");
  }
}
