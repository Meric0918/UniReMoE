# 编译论文

从任意工作目录执行：

```bash
/Users/bytedance/CodeBase/UniReMoE/paper/manuscript/build.sh
```

输出为同目录的 `main.pdf`，TeX 日志为 `main.log`。脚本优先使用
`CODEX_TECTONIC_PATH`、PATH 中或桌面应用自带的 Tectonic；若不可用，回退到
TeX Live／MacTeX 的 `pdflatex → bibtex → pdflatex → pdflatex`。
Tectonic 自带 BibTeX 处理能力，本稿不必因为使用 `\bibliography` 就安装完整 TeX Live。
首次编译可能需要联网下载宏包和字体。

## 本次编译失败原因及修复

- OUP 模板将 `\dag` 重定义成自带数学模式的文本上标。正文表格中的
  `$^{\dag}$` 会因此嵌套数学模式，产生 `Missing } inserted`。
  已改为数学符号 `\dagger`；双剑号使用 `\ddagger`。
- OUP 的印刷裁切钩子生成 PostScript `pdfmark`，在 XeTeX／Tectonic 的
  xdvipdfmx 后端会产生转换警告。`main.tex` 中增加了仅在 XeTeX 下启用的
  页面尺寸钩子；不修改发行版 class 文件，也不启用 shell escape。
- 扩展后的 Key Points 超出首页容量，现移到标题页之后；过高的相关工作对比表
  缩小字号，窄表中的长标签改为可换行列。论文内容及实验数值不变。

编译器自动选择信息中的 `bibliography tooling` 是启发式选择提示，
`No TeX Live or MacTeX installation detected` 是缺少备用工具链的提示。
应查看最终失败引擎的具体日志来定位源码错误。

本稿仍含作者信息、图示和结果的 TODO／N/A，生成 PDF 不代表已达到投稿状态。
现阶段仍可见模板及排版的 underfull／overfull 警告；它们与上述致命错误不同。
