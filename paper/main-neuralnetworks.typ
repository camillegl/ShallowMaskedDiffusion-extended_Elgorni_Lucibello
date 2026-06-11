#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "@preview/mannot:0.3.0": markrect

#show figure.caption: set align(left)
#show figure.caption: set text(style: "italic")

// #import "@preview/equate:0.3.2": equate

//// ALTERNATIVE ARTICLE TEMPLATES
// #import "@preview/rubber-article:0.5.0": article, appendix, maketitle
// #import "@preview/starter-journal-article:0.4.0": article, author-meta, appendix
// #import "@preview/elsearticle:0.4.2": *
//
#show: arkheion.with(
  title: [Masked Diffusion Models as Associative Memories],
  authors: (
    (name: "Filippo Elgorni", email: "filippo.elgorni@phd.unibocconi.it", affiliation: [Bocconi University, Milan]),
    (name: "Carlo Lucibello", email: "carlo.lucibello@unibocconi.it", affiliation: [Bocconi University, Milan]),
  ),
  date: datetime.today().display("[day] [month repr:Long] [year]"),
  abstract: [#align(left)[TODO]],
)

// #let note(content) = inline-note(content, fill: gray.lighten(80%), stroke: gray, par-break: false)
#let note(content) = highlight([NOTE: ] + content, fill: gray.lighten(80%))

#let citecolor = rgb("#93430e")
#show cite: set text(fill: citecolor)
#show link: set text(fill: blue)
#show link: underline
#show ref: set text(fill: blue)

// #show: equate.with(breakable: true, sub-numbering: true)
// #set math.equation(numbering: "(1.1)")

#let cM = $cal(M)$
#let ovx = $overline(x)$
#let tx = $tilde(x)$

#outline(depth: 2, indent: n => n * 1em)

= Introduction