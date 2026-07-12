# Render with:  vhs docs/demo.tape   ->   docs/demo.gif
# (install VHS: https://github.com/charmbracelet/vhs)
Output docs/demo.gif
Set FontSize 20
Set Width 1100
Set Height 640
Set Padding 18
Set Theme "Catppuccin Mocha"

Type "pip install raggate" Sleep 400ms Enter Sleep 2s
Type "raggate init" Sleep 400ms Enter Sleep 2s
Type "raggate run" Sleep 400ms Enter Sleep 3s
Comment "Tighten the faithfulness gate, then watch it FAIL (exit 1) — the money shot"
Type "raggate gate --gates evals/gates.high-stakes.yaml" Sleep 400ms Enter Sleep 3s
Sleep 2s
