const fs = require('fs');

const md = fs.readFileSync('Research_Paper_Industry_Specific_Conversational_AI.md', 'utf-8');

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderMarkdown(content) {
  let lines = content.split('\n');
  let html = [];
  let inCode = false;
  let codeLang = '';
  let codeBuffer = [];
  let inTable = false;
  let tableBuffer = [];

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Code blocks
    if (line.startsWith('```')) {
      if (!inCode) {
        inCode = true;
        codeLang = line.replace('```', '').trim();
        codeBuffer = [];
        continue;
      } else {
        inCode = false;
        html.push(`<pre><code class="language-${codeLang}">${escapeHtml(codeBuffer.join('\n'))}</code></pre>`);
        continue;
      }
    }
    if (inCode) {
      codeBuffer.push(line);
      continue;
    }

    // Tables
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableBuffer = [];
      }
      tableBuffer.push(line.trim());
      continue;
    } else if (inTable) {
      inTable = false;
      html.push(renderTable(tableBuffer));
      tableBuffer = [];
    }

    // Headers
    if (line.startsWith('# ')) {
      html.push(`<h1>${formatInline(line.substring(2))}</h1>`);
      continue;
    }
    if (line.startsWith('## ')) {
      html.push(`<h2>${formatInline(line.substring(3))}</h2>`);
      continue;
    }
    if (line.startsWith('### ')) {
      html.push(`<h3>${formatInline(line.substring(4))}</h3>`);
      continue;
    }
    if (line.startsWith('#### ')) {
      html.push(`<h4>${formatInline(line.substring(5))}</h4>`);
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      html.push(`<blockquote><p>${formatInline(line.substring(2))}</p></blockquote>`);
      continue;
    }

    // Horizontal Rule
    if (line.trim() === '---') {
      html.push('<hr/>');
      continue;
    }

    // Ordered / Unordered lists
    if (line.match(/^\s*[-*]\s+/)) {
      html.push(`<li>${formatInline(line.replace(/^\s*[-*]\s+/, ''))}</li>`);
      continue;
    }
    if (line.match(/^\s*\d+\.\s+/)) {
      html.push(`<li>${formatInline(line.replace(/^\s*\d+\.\s+/, ''))}</li>`);
      continue;
    }

    // Empty lines
    if (!line.trim()) {
      continue;
    }

    // Standard paragraph
    html.push(`<p>${formatInline(line)}</p>`);
  }

  if (inTable && tableBuffer.length > 0) {
    html.push(renderTable(tableBuffer));
  }

  return html.join('\n');
}

function renderTable(tableLines) {
  if (tableLines.length < 2) return '';
  let out = '<table>\n<thead>\n<tr>\n';
  let headers = tableLines[0].split('|').map(s => s.trim()).filter((s, idx, arr) => idx > 0 && idx < arr.length - 1);
  for (let h of headers) {
    out += `  <th>${formatInline(h)}</th>\n`;
  }
  out += '</tr>\n</thead>\n<tbody>\n';
  
  for (let r = 2; r < tableLines.length; r++) {
    let cells = tableLines[r].split('|').map(s => s.trim()).filter((s, idx, arr) => idx > 0 && idx < arr.length - 1);
    out += '<tr>\n';
    for (let c of cells) {
      out += `  <td>${formatInline(c)}</td>\n`;
    }
    out += '</tr>\n';
  }
  out += '</tbody>\n</table>\n';
  return out;
}

function formatInline(str) {
  // Bold italic
  str = str.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
  // Bold
  str = str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Italic
  str = str.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Inline code
  str = str.replace(/`([^`]+)`/g, '<code>$1</code>');
  return str;
}

const bodyHtml = renderMarkdown(md);

const completeDoc = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Architecting Domain-Specialized Large Language Models for Enterprise IT Service Management</title>
  <style>
    @page {
      size: letter;
      margin: 1in;
      @bottom-right {
        content: counter(page);
      }
    }
    body {
      font-family: "Times New Roman", Times, Georgia, serif;
      font-size: 11pt;
      line-height: 1.5;
      color: #111;
      max-width: 8.5in;
      margin: 0 auto;
      padding: 0.8in 1in;
      background-color: #fff;
    }
    h1 {
      font-size: 20pt;
      line-height: 1.25;
      text-align: center;
      margin-bottom: 1.5rem;
      font-weight: bold;
    }
    h2 {
      font-size: 14pt;
      border-bottom: 1.5px solid #222;
      padding-bottom: 4px;
      margin-top: 2rem;
      margin-bottom: 0.8rem;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    h3 {
      font-size: 12pt;
      margin-top: 1.4rem;
      margin-bottom: 0.5rem;
      font-weight: bold;
    }
    h4 {
      font-size: 11pt;
      font-style: italic;
      margin-top: 1rem;
      margin-bottom: 0.3rem;
    }
    p {
      text-align: justify;
      margin-bottom: 0.9rem;
      text-indent: 1.5em;
    }
    h1 + p, h2 + p, h3 + p, h4 + p, table + p, pre + p {
      text-indent: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.2rem 0;
      font-size: 9.5pt;
    }
    th, td {
      border: 1px solid #666;
      padding: 6px 10px;
      text-align: left;
    }
    th {
      background-color: #f2f2f2;
      font-weight: bold;
    }
    pre {
      background-color: #f8f9fa;
      border: 1px solid #dcdcdc;
      padding: 10px;
      border-radius: 4px;
      font-family: "Consolas", "Courier New", monospace;
      font-size: 9pt;
      line-height: 1.35;
      overflow-x: auto;
      margin: 1rem 0;
    }
    code {
      font-family: "Consolas", "Courier New", monospace;
      font-size: 9.5pt;
      background-color: #f3f3f3;
      padding: 1px 4px;
      border-radius: 2px;
    }
    blockquote {
      border-left: 3px solid #0056b3;
      margin: 1rem 0;
      padding: 0.5rem 1rem;
      background-color: #f4f8fb;
      font-style: italic;
    }
    hr {
      border: 0;
      border-top: 1px solid #ccc;
      margin: 1.5rem 0;
    }
    li {
      margin-bottom: 0.3rem;
    }
    @media print {
      body {
        padding: 0;
        max-width: 100%;
      }
      h2 {
        page-break-before: auto;
      }
      table, pre, blockquote {
        page-break-inside: avoid;
      }
    }
  </style>
</head>
<body>
${bodyHtml}
</body>
</html>`;

fs.writeFileSync('Research_Paper_Industry_Specific_Conversational_AI.html', completeDoc);
console.log('HTML Research Paper generated successfully: Research_Paper_Industry_Specific_Conversational_AI.html');
