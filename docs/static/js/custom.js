document.addEventListener('DOMContentLoaded', () => {

  // open external links of marked sections in a new tab
  document.querySelectorAll('.new-tab a.reference.external').forEach((a) => {
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
  });

  // show example of a method inline, under its table row,
  // toggled in the usage column,
  // without JavaScript, the examples stay listed below the table
  document.querySelectorAll('table.method-table tbody tr').forEach((row) => {
    const name = row.cells[0];
    const key = name.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const example = document.getElementById(`example-${key}`);
    if (!example) return;

    // hidden row spanning the table, holding the example
    const detail = document.createElement('tr');
    detail.className = 'example-row';
    detail.hidden = true;
    const cell = document.createElement('td');
    cell.colSpan = row.cells.length;
    cell.append(example);
    detail.append(cell);
    row.after(detail);

    // toggle in the usage column
    const link = document.createElement('a');
    link.href = `#example-${key}`;
    link.className = 'example-toggle';
    link.title = 'Show example';
    link.setAttribute('aria-label', `Example of ${name.textContent.trim()}`);
    link.setAttribute('aria-expanded', 'false');
    // module of the method, e.g., cl.hill, as in the example
    const call = example.textContent.match(/cl\.\w+/);
    const code = document.createElement('code');
    code.textContent = call ? call[0] : 'example';
    const arrow = document.createElement('span');
    arrow.className = 'example-arrow';
    arrow.textContent = '\u25B8';
    link.append(arrow, code);
    link.addEventListener('click', (event) => {
      event.preventDefault();
      detail.hidden = !detail.hidden;
      link.setAttribute('aria-expanded', String(!detail.hidden));
      link.title = detail.hidden ? 'Show example' : 'Hide example';
      arrow.textContent = detail.hidden ? '\u25B8' : '\u25BE';
    });
    row.cells[2].replaceChildren(link);

    // open, if linked directly
    if (location.hash === `#example-${key}`) link.click();
  });
});
