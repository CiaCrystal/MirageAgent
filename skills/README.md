# Migration skills

每个目标包的 `skills/` 保存按主题组织的迁移指导。Markdown 文件以 `description` front matter 描述适用场景，正文包含旧写法、新写法及注意事项；`INDEX.md` 是目录说明。

默认开启按需 skills：系统提示展示 skill 目录，Agent 通过 `load_skill` 按需加载正文。设置 `MIGRATION_AGENT_SKILLS_EXTERNAL=false` 可关闭，改用目标包的简短速查提示。
