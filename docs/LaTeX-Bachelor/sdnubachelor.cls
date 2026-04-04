%%
%% This is file `sdnubachelor.cls',
%%
%%
%% This file may be distributed and/or modified under the conditions of
%% the LaTeX Project Public License, either version 1.2 of this license
%% or (at your option) any later version.  The latest version of this
%% license is in:
%%
%%    http://www.latex-project.org/lppl.txt
%%
%% and version 1.2 or later is part of all distributions of LaTeX version
%% 1999/12/01 or later.
%%
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{sdnubachelor}[2026/03/05 v2.0 SDNU Bachelor Thesis 2026 Format]

\PassOptionsToClass{AutoFakeBold}{ctexart}
\LoadClassWithOptions{ctexart}


% 2026: A4, 上/左2.5cm 下/右2cm; 页眉边距1.5cm, 页脚边距1.75cm
\RequirePackage[a4paper,top=2.5cm,bottom=2cm,left=2.5cm,right=2cm,
	includehead,includefoot,headheight=14pt,headsep=1.5cm,footskip=1.75cm]{geometry}

\RequirePackage{ifthen}
\DeclareRobustCommand{\sdnuprint}{}
\DeclareRobustCommand{\sdnupaperprint}[1]{
	\renewcommand{\sdnuprint}{#1}
}

\DeclareRobustCommand{\sdnuseccenter}{}
\DeclareRobustCommand{\sdnusectioncenter}[1]{
	\renewcommand{\sdnuseccenter}{#1}
}

\pagestyle{plain}

\RequirePackage{tabu}
\RequirePackage{makecell}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{bm}
\usepackage{gbt7714}

\DeclareRobustCommand{\xiaosi}{\zihao{-4}}
\DeclareRobustCommand{\sihao}{\zihao{4}}  

\RequirePackage{algorithmic}
\RequirePackage{algorithm}

\floatname{algorithm}{算法}  
\renewcommand{\algorithmicrequire}{\textbf{输入:}}  
\renewcommand{\algorithmicensure}{\textbf{输出:}} 

\RequirePackage{setspace}
\RequirePackage{titlesec}
% 2026: 一级 黑体三号加粗居中 段前24pt 段后18pt
\titleformat{\section}
{\centering\heiti\zihao{3}\bfseries}
{\thesection \quad}
{0pt}
{}

% 2026: 二级 黑体小三加粗左对齐 段前24pt 段后6pt
\titleformat{\subsection}
{\raggedright\heiti\zihao{-3}\bfseries}
{\thesubsection \quad}
{0pt}
{}

% 2026: 三级 黑体四号加粗两端对齐 段前12pt 段后6pt
\titleformat{\subsubsection}
{\raggedright\heiti\zihao{4}\bfseries}
{\thesubsubsection \quad}
{0pt}
{}

% 2026: 四级 黑体小四加粗两端对齐 段前6pt 段后6pt
\setcounter{secnumdepth}{4}
\renewcommand{\theparagraph}{\thesubsubsection.\arabic{paragraph}}
\titleformat{\paragraph}[runin]
{\heiti\zihao{-4}\bfseries}
{\theparagraph\quad}
{0pt}
{}

\titlespacing{\section}{0pt}{24pt}{18pt}
\titlespacing{\subsection}{0pt}{24pt}{6pt}
\titlespacing{\subsubsection}{0pt}{12pt}{6pt}
\titlespacing{\paragraph}{0pt}{6pt}{6pt}

\RequirePackage{enumitem}
\setenumerate[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=5pt}
\setitemize[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=5pt}
\setdescription{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=5pt}

\RequirePackage{amsmath}
\numberwithin{figure}{section}
\numberwithin{table}{section}
\renewcommand {\thefigure} {\thesection{}-\arabic{figure}}
\renewcommand {\thetable} {\thesection{}-\arabic{table}}

\newsavebox\CBox
\def\textBF#1{\sbox\CBox{#1}\resizebox{\wd\CBox}{\ht\CBox}{\textbf{#1}}}

% 2026: 页眉 山东师范大学学士学位论文 宋体五号居中; 页脚 页码 TNR五号居中
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\songti\zihao{5} 山东师范大学学士学位论文}
\fancyfoot[C]{\zihao{5}\textrm{\thepage}}

% 2026: 目录标题 黑体三号加粗居中 段前24pt 段后18pt; 目录项 小四 固定25pt
\RequirePackage{titletoc}
\hypersetup{hidelinks}
\renewcommand{\contentsname}{\vspace*{24pt}\begin{center}\heiti\zihao{3}\bfseries 目~~录\end{center}\vspace*{18pt}}
\titlecontents{section}[1.2em]
{\vspace*{0mm}\zihao{-4}\setlength{\baselineskip}{25pt}}
{\contentslabel{1em}}
{\hspace*{-1.2em}}
{\titlerule*[0.5pc]{$\cdot$}\contentspage}
\titlecontents{subsection}[4.2em]
{\vspace*{0mm}\zihao{-4}\setlength{\baselineskip}{25pt}}
{\contentslabel{2em}}
{}
{\titlerule*[0.5pc]{$\cdot$}\contentspage}
\titlecontents{subsubsection}[7.2em]
{\vspace*{0mm}\zihao{-4}\setlength{\baselineskip}{25pt}}
{\contentslabel{2.5em}}
{}
{\titlerule*[0.5pc]{$\cdot$}\contentspage}
\titlecontents{paragraph}[10.2em]
{\vspace*{0mm}\zihao{-4}\setlength{\baselineskip}{25pt}}
{\contentslabel{2.5em}}
{}
{\titlerule*[0.5pc]{$\cdot$}\contentspage}

\newcommand{\upcite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}
\RequirePackage{graphicx}
\graphicspath{{data/img/}}
\RequirePackage{subfig}
\RequirePackage{placeins}
% 2026: 图题/表题 宋体(TNR) 五号 居中 单倍行距
\RequirePackage{caption}
\DeclareCaptionFont{wuhao}{\zihao{5}}
\captionsetup{font=wuhao, justification=centering, labelsep=space}

% 双语图表标题 (中文/英文)：图/Fig.，表/Tab.
\RequirePackage{bicaption}
% 统一中英文双语标题字体与间距
\bicaptionsetup{font=wuhao, labelsep=space}{font=wuhao, labelsep=space}
% figure: 中文“图”，英文“Fig.”
\bicaptionsetup[figure]{name=图}{name=Fig.}
% table: 中文“表”，英文“Tab.”
\bicaptionsetup[table]{name=表}{name=Tab.}

\RequirePackage{amsthm}
\newtheoremstyle{ctstyle}{2pt}{2pt}{\kaishu}{2em}{\bfseries}{.}{1em}{}
\theoremstyle{ctstyle}
\newtheorem{theorem}{定理}
\newtheorem{definition}{定义}

\renewcommand{\proofname}{证明}
%\renewcommand*{\qedsymbol}{[证毕]}
\makeatletter
\renewenvironment{proof}[1][\proofname]{\par
	\pushQED{\qed}%
	\normalfont \topsep0\p@\@plus6\p@\labelsep1em\relax
	\trivlist
	\item[\hskip\labelsep\indent
	\bfseries
	#1\@addpunct{.}]\ignorespaces
}{%
	\popQED\endtrivlist\@endpefalse
}
\makeatother

% 正文全局: 首行缩进2字符 (2026)
\setlength{\parindent}{2em}

\ctexset{
	bibname = 参考文献,
}

\RequirePackage{tabularx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{amsmath}
\RequirePackage{listings}


\lstset{
	language=C,
	basicstyle=\small\ttfamily,
	numbers=left,
	numbersep=5pt,
	xleftmargin=20pt,
	frame=tb,
	framexleftmargin=20pt
}
\renewcommand\lstlistingname{代码}
\renewcommand*\thelstnumber{\arabic{lstnumber}:}
\DeclareCaptionFormat{mylst}{\hrule#1#2#3}
\captionsetup[lstlisting]{format=mylst,labelfont=bf,singlelinecheck=off,labelsep=space}

\DeclareRobustCommand{\sdnutitlechs}{}
\DeclareRobustCommand{\sdnutitleen}{}
\DeclareRobustCommand{\sdnutitle}[2]{
	\renewcommand{\sdnutitlechs}{#1}
	\renewcommand{\sdnutitleen}{#2}
}

\DeclareRobustCommand{\sdnuauthorchs}{}
\DeclareRobustCommand{\sdnuauthorens}{}
\DeclareRobustCommand{\sdnuauthorid}{}
\DeclareRobustCommand{\sdnuauthor}[3]{
	\renewcommand{\sdnuauthorchs}{#1}
	\renewcommand{\sdnuauthorens}{#2}
	\renewcommand{\sdnuauthorid}{#3}
}

\DeclareRobustCommand{\sdnumentorchs}{}
\DeclareRobustCommand{\sdnumentor}[1]{
	\renewcommand{\sdnumentorchs}{#1}
}

\DeclareRobustCommand{\sdnucollegechs}{}
\DeclareRobustCommand{\sdnumajorchs}{}
\DeclareRobustCommand{\sdnuclasschs}{}
\DeclareRobustCommand{\sdnuinfo}[3]{
	\renewcommand{\sdnucollegechs}{#1}
	\renewcommand{\sdnumajorchs}{#2}
	\renewcommand{\sdnuclasschs}{#3}
}

\DeclareRobustCommand{\sdnuyear}{}
\DeclareRobustCommand{\sdnumon}{}
\DeclareRobustCommand{\sdnuday}{}
\DeclareRobustCommand{\sdnudate}[3]{
	\renewcommand{\sdnuyear}{#1}
	\renewcommand{\sdnumon}{#2}
	\renewcommand{\sdnuday}{#3}
}

\DeclareRobustCommand{\sdnukeywchs}{}
\DeclareRobustCommand{\sdnukeywen}{}
\DeclareRobustCommand{\sdnukeyw}[2]{
	\renewcommand{\sdnukeywchs}{#1}
	\renewcommand{\sdnukeywen}{#2}
}

\DeclareRobustCommand{\classification}{}
\DeclareRobustCommand{\sdnuclassification}[1]{
	\renewcommand{\classification}{#1}
}

\DeclareRobustCommand{\sdnuen}{}
\DeclareRobustCommand{\sdnucollegeen}{}
\DeclareRobustCommand{\sdnuinfoen}[2]{
	\renewcommand{\sdnuen}{#1}
	\renewcommand{\sdnucollegeen}{#2}
}



%%
%% End of file `sdnubachelor.cls'.
