/**
 * Enhanced Code Highlighting for Stoolap Documentation
 * 
 * This script enhances code highlighting for SQL and Go code blocks
 * by adding language badges and improving syntax highlighting.
 */

document.addEventListener('DOMContentLoaded', function() {
  // Mark code blocks with their language
  document.querySelectorAll('div.highlighter-rouge').forEach(function(block) {
    // Extract language from class names
    const classes = block.className.split(' ');
    for (let i = 0; i < classes.length; i++) {
      if (classes[i].startsWith('language-')) {
        const lang = classes[i].replace('language-', '');
        block.setAttribute('data-lang', lang.toUpperCase());
        break;
      }
    }
  });

  // Enhance SQL code blocks
  document.querySelectorAll('div.language-sql pre.highlight code').forEach(function(code) {
    // Find SQL keywords and apply special styling
    const sqlKeywords = [
      'SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 
      'ON', 'AND', 'OR', 'IN', 'NOT', 'IS', 'NULL', 'AS', 'ORDER', 'BY', 
      'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'INSERT', 'INTO', 'VALUES', 
      'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE', 'INDEX', 'ALTER', 'DROP',
      'CONSTRAINT', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'UNIQUE', 
      'DEFAULT', 'CASCADE', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
    ];

    // Add table name styling
    let html = code.innerHTML;
    
    // Mark SQL table names
    const tablePattern = /\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g;
    html = html.replace(tablePattern, function(match, keyword, tableName) {
      return keyword + ' <span class="table-name">' + tableName + '</span>';
    });

    // Mark SQL aliases
    const aliasPattern = /\b(AS)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g;
    html = html.replace(aliasPattern, function(match, keyword, alias) {
      return keyword + ' <span class="alias">' + alias + '</span>';
    });

    code.innerHTML = html;
  });

  // Enhance Go code blocks
  document.querySelectorAll('div.language-go pre.highlight code').forEach(function(code) {
    // Add specific Go styling if needed
    // This is mostly handled by CSS, but we could enhance it further here
  });

  // Copy button removed as requested
});