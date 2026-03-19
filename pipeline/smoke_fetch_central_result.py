from fetch_central_result import parse_result_page


HTML_SAMPLE = """
<html>
  <body>
    <table>
      <tbody>
        <tr class="FirstDisplay HorseList">
          <td class="Result_Num"><div class="Rank">1</div></td>
          <td class="Num Txt_C"><div>1</div></td>
          <td class="Horse_Info">
            <span class="Horse_Name"><a><span class="HorseNameSpan">インパクトシー</span></a></span>
          </td>
        </tr>
        <tr class="FirstDisplay HorseList">
          <td class="Result_Num"><div class="Rank">2</div></td>
          <td class="Num Txt_C"><div>7</div></td>
          <td class="Horse_Info">
            <span class="Horse_Name"><a><span class="HorseNameSpan">サンプル二着</span></a></span>
          </td>
        </tr>
        <tr class="FirstDisplay HorseList">
          <td class="Result_Num"><div class="Rank">3</div></td>
          <td class="Num Txt_C"><div>8</div></td>
          <td class="Horse_Info">
            <span class="Horse_Name"><a><span class="HorseNameSpan">サンプル三着</span></a></span>
          </td>
        </tr>
      </tbody>
    </table>
    <div class="ResultPaybackLeftWrap">
      <table class="Payout_Detail_Table">
        <tbody>
          <tr class="Tansho">
            <th>単勝</th>
            <td class="Result">
              <div><span>1</span></div>
            </td>
            <td class="Payout"><span>270円</span></td>
            <td class="Ninki"><span>2人気</span></td>
          </tr>
          <tr class="Fukusho">
            <th>複勝</th>
            <td class="Result">
              <div><span>1</span></div>
              <div><span>7</span></div>
              <div><span>8</span></div>
            </td>
            <td class="Payout"><span>100円<br/>100円<br/>100円</span></td>
            <td class="Ninki"><span>1人気</span><span>2人気</span><span>3人気</span></td>
          </tr>
        </tbody>
      </table>
      <table class="Payout_Detail_Table">
        <tbody>
          <tr class="Wide">
            <th>ワイド</th>
            <td class="Result">
              <ul><li><span>1</span></li><li><span>7</span></li><li></li></ul>
              <ul><li><span>1</span></li><li><span>8</span></li><li></li></ul>
              <ul><li><span>7</span></li><li><span>8</span></li><li></li></ul>
            </td>
            <td class="Payout"><span>200円<br/>150円<br/>130円</span></td>
            <td class="Ninki"><span>3人気</span><span>2人気</span><span>1人気</span></td>
          </tr>
          <tr class="Tan3">
            <th>3連単</th>
            <td class="Result">
              <ul><li><span>1</span></li><li><span>7</span></li><li><span>8</span></li></ul>
            </td>
            <td class="Payout"><span>2,000円</span></td>
            <td class="Ninki"><span>6人気</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </body>
</html>
"""


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    payload = parse_result_page(
        HTML_SAMPLE,
        source_url="https://race.netkeiba.com/race/result.html?race_id=202606020608&rf=race_submenu",
    )
    assert_true(payload["race_id"] == "202606020608", "race_id parse failed")
    assert_true(len(payload["top3"]) == 3, "top3 length failed")
    assert_true(payload["top3"][0]["horse_no"] == "1", "top1 horse_no failed")
    assert_true(payload["top3"][0]["horse_name"] == "インパクトシー", "top1 horse_name failed")
    assert_true(payload["payouts"]["単勝"][0]["payout_yen"] == 270, "tansho payout failed")
    assert_true(payload["payouts"]["複勝"][1]["horse_numbers"] == ["7"], "fukusho parse failed")
    assert_true(payload["payouts"]["ワイド"][2]["horse_numbers"] == ["7", "8"], "wide parse failed")
    assert_true(payload["payouts"]["3連単"][0]["horse_numbers"] == ["1", "7", "8"], "trifecta parse failed")
    print("smoke_fetch_central_result: OK")


if __name__ == "__main__":
    main()
