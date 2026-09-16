param([Parameter(Mandatory = $true)][string]$WorkbookPath)

$ErrorActionPreference = 'Stop'
$expected = Join-Path $PSScriptRoot 'country_performance.xlsx'
if ((Resolve-Path -LiteralPath $WorkbookPath).Path -ne (Resolve-Path -LiteralPath $expected).Path) {
    throw 'Only the country-performance workbook may be recalculated.'
}
$excel = $null
$book = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.EnableEvents = $false
    $excel.AutomationSecurity = 3
    $book = $excel.Workbooks.Open($expected, 0, $false)
    $excel.CalculateFullRebuild()
    $book.Save()
    Write-Output 'Excel recalculation and save complete.'
}
finally {
    try {
        if ($null -ne $book) {
            $book.Close($false)
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($book)
        }
    }
    finally {
        if ($null -ne $excel) {
            $excel.Quit()
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel)
        }
    }
}
