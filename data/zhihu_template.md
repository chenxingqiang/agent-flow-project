---
title: "${title}"
author: "XinJian Chen"
output: 
  pdf_document:
    latex_engine: xelatex
    keep_tex: true
header-includes:
  - \usepackage{fontspec}
  - \usepackage{unicode-math}
  - \setmathfont{XITS Math}
  - \setmainfont{Times New Roman}
  - \setmonofont{Monaco}
  - \usepackage{fontspec}
  - \usepackage{xeCJK}
  - \usepackage{hyperref}
  - \usepackage{graphicx}
  - \usepackage{titlesec}
  - \usepackage{enumitem}
  - \usepackage{fancyhdr}
  - \usepackage{lastpage}
  - \usepackage{xcolor}
  - \usepackage{setspace}
  - \usepackage{titling}
  - \usepackage{tikz}
  - \usepackage[most]{tcolorbox}
  - \usepackage{draftwatermark}
  - \SetWatermarkText{Confidential}
  - \SetWatermarkScale{0.5}
  - \SetWatermarkColor[gray]{0.95}
  - \setmonofont{Monaco}
  - \definecolor{accent}{RGB}{0,90,160}
  - \definecolor{lightaccent}{RGB}{230,240,250}
  - \definecolor{titlecolor}{RGB}{0,90,160}
  - \definecolor{linkcolor}{RGB}{0,0,255}
  - \definecolor{citecolor}{RGB}{0,128,0}
  - \definecolor{urlcolor}{RGB}{128,0,128}
  - \hypersetup{colorlinks=true, linkcolor=linkcolor, citecolor=citecolor, urlcolor=urlcolor}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyfoot[R]{\color{accent}\small Page \thepage\ of \pageref{LastPage}}
  - \fancyhead[L]{\includegraphics[height=0.5cm]{zhihu-logo.png}}
  - \fancyhead[C]{\color{blue} 知乎学院 - Leading AI for Science Solutions}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \renewcommand{\footrulewidth}{0.4pt}
  - \setstretch{1.5}
geometry: margin=1in



---


\begin{titlepage}
\begin{center}
\vspace*{\fill}

% Replace with your actual photo path
\includegraphics[width=0.5\textwidth]{zhihu-logo.png}

\vspace{2cm}

{\huge\bfseries\color{titlecolor} 基于静态图的恶意软件分类方法研究}

\vspace{1cm}

{\Large Dr. XinJian Chen}

\vspace{0.5cm}

{\large Artificial Intelligence Specialist, PhD}

\vspace{1cm}

{\large December 4, 2024}

\vspace*{\fill}
\end{center}
\end{titlepage}

\newpage
\tableofcontents
\newpage

# Personal Profile

I am currently a Senior AI Compiler Developer at a leading computing and storage company, with 7 years of frontline experience in algorithm research and development. I specialize in deep learning algorithms, engineering, product design, low-level optimization, and AI chip compiler development. I have previously worked at top companies such as Bianlifeng, Baidu (Baidu Biotech), and Blue Elephant, where I focused on privacy computing, machine learning, and AI.

- **Languages**: Python, C++, Java, Scala, Go, JavaScript
- **Frameworks**: Hadoop, Flink, Kubernetes, TensorFlow, PyTorch
- **Deep Learning**: Transformer, BERT, CNN, RL, GNN
- **Specialization**: Privacy Computing, Federated Learning, MPC, Machine Learning Optimization

\newpage

# ${title}

\newpage
