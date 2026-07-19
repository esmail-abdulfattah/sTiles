/**
 * sTiles download-form backend (Google Apps Script)
 * -------------------------------------------------
 * Receives POSTs from docs/download.html, appends one row per submission to a
 * Google Sheet, and emails you a notification.
 *
 * SETUP (one time, ~3 minutes) — see README.md in this folder for screenshots-level detail:
 *   1. Create a Google Sheet (any name). Copy its ID from the URL:
 *        https://docs.google.com/spreadsheets/d/<THIS_IS_THE_ID>/edit
 *   2. In that Sheet: Extensions > Apps Script. Delete the sample code,
 *      paste THIS file, and set the two constants below.
 *   3. Deploy > New deployment > type "Web app".
 *        - Execute as:  Me
 *        - Who has access:  Anyone
 *      Copy the "Web app" URL it gives you (ends in /exec).
 *   4. Paste that URL into APPS_SCRIPT_URL in docs/download.html.
 *
 * Re-deploy note: after editing this script, use Deploy > Manage deployments >
 * (edit, new version) so the /exec URL keeps working, or the changes won't go live.
 */

// ---- EDIT THESE TWO ----
const SHEET_ID     = 'PASTE_YOUR_GOOGLE_SHEET_ID_HERE';
const NOTIFY_EMAIL = 'esmailabdulfattah@gmail.com';   // where notifications go
// ------------------------

const SHEET_NAME = 'Downloads';
const HEADERS = ['Timestamp', 'Name', 'Email', 'Country', 'Institution',
                 'Primary use', 'Reason', 'Page', 'User agent'];

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const sheet = getSheet_();
    sheet.appendRow([
      new Date(),
      data.name        || '',
      data.email       || '',
      data.country     || '',
      data.institution || '',
      data.usetype     || '',
      data.reason      || '',
      data.page        || '',
      data.userAgent   || ''
    ]);

    notify_(data);
    return json_({ result: 'ok' });
  } catch (err) {
    return json_({ result: 'error', message: String(err) });
  }
}

// Lets you open the /exec URL in a browser to confirm it is live.
function doGet() {
  return json_({ result: 'ok', service: 'sTiles download form', time: new Date() });
}

function getSheet_() {
  const ss = SpreadsheetApp.openById(SHEET_ID);
  let sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
  }
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADERS);
    sheet.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function notify_(data) {
  if (!NOTIFY_EMAIL) return;
  const subject = 'sTiles download: ' + (data.name || 'unknown') +
                  ' (' + (data.institution || 'n/a') + ')';
  const body =
    'New sTiles binary download request\n\n' +
    'Name:        ' + (data.name || '') + '\n' +
    'Email:       ' + (data.email || '') + '\n' +
    'Country:     ' + (data.country || '') + '\n' +
    'Institution: ' + (data.institution || '') + '\n' +
    'Primary use: ' + (data.usetype || '') + '\n' +
    'Reason:      ' + (data.reason || '') + '\n\n' +
    'Page:        ' + (data.page || '') + '\n' +
    'User agent:  ' + (data.userAgent || '') + '\n';
  try {
    MailApp.sendEmail(NOTIFY_EMAIL, subject, body);
  } catch (err) {
    // Non-fatal: the row is already saved even if the email quota is hit.
  }
}

function json_(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
