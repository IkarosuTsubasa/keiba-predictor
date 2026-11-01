"""Export official course dimensions and digitized elevation diagrams for the demo.

Only writes frontend fixtures. Reference downloads remain in analysis_outputs.
The chart trace and plan control points are approximations, not survey/GIS data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keiba_llm_agent.analysis.course_geometry import _COURSE_PROFILES
from keiba_llm_agent.analysis.local_course_geometry import _LOCAL_COURSE_PROFILES

SLUGS = dict(zip(
    ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀"],
    ["sapporo", "hakodate", "fukushima", "niigata", "tokyo", "nakayama", "chukyo", "kyoto", "hanshin", "kokura", "monbetsu", "morioka", "mizusawa", "urawa", "funabashi", "oi", "kawasaki", "kanazawa", "kasamatsu", "nagoya", "sonoda", "himeji", "kochi", "saga"],
))
LABELS = {"standard": "標準", "inner": "内回り", "outer": "外回り", "straight": "直線", "outer_right": "外・右回り", "inner_right": "内・右回り", "outer_left": "外・左回り"}

# Coarse control points read from each official plan, starting at the finish in
# the direction of travel. Branch lines and temporary rail positions are omitted.
PLANS = {
    "sapporo": [[320,355],[230,355],[145,325],[105,255],[105,175],[150,105],[230,70],[625,70],[710,105],[755,175],[755,255],[710,325],[625,355]],
    "hakodate": [[360,350],[235,350],[140,320],[95,260],[90,175],[135,110],[225,80],[640,80],[730,110],[775,175],[770,260],[725,320],[640,350]],
    "fukushima": [[320,355],[210,355],[120,320],[80,250],[90,170],[145,105],[230,80],[620,80],[715,115],[775,180],[780,255],[735,320],[650,355]],
    "niigata:outer": [[620,350],[710,350],[775,315],[800,240],[790,155],[750,105],[670,85],[220,85],[125,105],[75,170],[70,255],[115,325],[200,350]],
    "niigata:inner": [[620,340],[710,340],[770,310],[790,235],[770,160],[720,120],[640,110],[390,110],[315,140],[285,210],[300,285],[355,330]],
    "niigata:dirt": [[620,330],[705,330],[755,300],[775,235],[755,170],[710,132],[635,125],[395,125],[330,152],[305,215],[320,280],[370,320]],
    "tokyo": [[610,360],[705,340],[755,290],[775,210],[755,140],[705,95],[620,85],[215,85],[135,110],[90,180],[85,265],[125,330],[205,365]],
    "nakayama:inner": [[330,370],[230,370],[150,335],[125,265],[150,195],[225,155],[640,155],[725,190],[775,260],[750,330],[680,370]],
    "nakayama:outer": [[330,370],[230,370],[145,340],[110,280],[150,210],[200,100],[255,55],[345,65],[500,105],[650,170],[760,240],[790,300],[750,350],[660,370]],
    "kyoto": [[330,365],[240,365],[170,335],[140,275],[145,200],[190,145],[260,120],[650,65],[715,85],[755,145],[770,235],[755,310],[710,355],[650,365]],
    "hanshin:inner": [[300,365],[230,365],[160,340],[130,285],[135,225],[180,180],[270,145],[590,80],[655,105],[700,170],[725,250],[705,310],[655,350],[600,365]],
    "hanshin:outer": [[300,365],[230,365],[160,340],[130,285],[135,225],[180,180],[270,145],[670,45],[735,65],[780,130],[800,215],[785,290],[745,345],[680,365]],
    "chukyo": [[590,360],[650,360],[710,335],[755,275],[755,205],[710,140],[635,110],[225,75],[145,115],[100,180],[95,265],[140,330],[225,360]],
    "kokura": [[350,360],[235,360],[140,330],[95,265],[90,185],[135,115],[220,80],[650,80],[730,110],[775,175],[780,255],[735,325],[650,360]],
    "monbetsu": [[320,355],[220,355],[135,325],[100,260],[105,185],[150,125],[230,95],[650,95],[730,120],[775,185],[775,265],[730,325],[650,355]],
    "morioka": [[530,355],[650,355],[735,320],[775,250],[770,175],[720,120],[640,100],[220,100],[140,130],[95,195],[100,275],[150,330],[225,355]],
    "mizusawa": [[350,355],[230,355],[145,320],[110,250],[115,175],[160,120],[240,95],[640,95],[720,125],[760,185],[760,265],[715,325],[640,355]],
    "urawa": [[540,355],[660,355],[740,320],[780,250],[775,175],[730,120],[650,95],[220,95],[145,125],[105,190],[110,270],[155,330],[230,355]],
    "funabashi": [[610,355],[690,340],[745,295],[775,215],[765,155],[720,115],[640,95],[235,95],[170,120],[130,180],[105,260],[145,325],[220,355]],
    "oi": [[330,355],[220,355],[150,325],[100,265],[125,205],[185,145],[250,105],[640,105],[720,135],[765,195],[775,265],[735,325],[650,355]],
    "kawasaki": [[580,355],[670,355],[750,320],[785,250],[775,170],[730,115],[650,90],[220,90],[140,120],[100,185],[100,265],[145,325],[225,355]],
    "kanazawa": [[350,355],[230,355],[150,325],[110,260],[110,180],[155,120],[235,95],[635,95],[715,125],[760,185],[765,265],[720,325],[640,355]],
    "kasamatsu": [[330,355],[220,355],[140,325],[105,260],[110,185],[155,125],[230,95],[640,95],[720,125],[765,190],[765,265],[720,325],[640,355]],
    "nagoya": [[350,355],[230,355],[140,325],[100,260],[110,185],[155,125],[235,95],[630,95],[715,125],[760,185],[765,265],[715,325],[635,355]],
    "sonoda": [[350,355],[230,355],[145,320],[110,255],[115,185],[165,125],[240,95],[630,95],[715,125],[755,190],[760,265],[710,325],[635,355]],
    "himeji": [[350,355],[225,355],[145,325],[105,260],[110,180],[155,120],[235,95],[640,95],[720,125],[760,185],[760,265],[715,325],[640,355]],
    "kochi": [[320,350],[235,335],[150,305],[100,255],[125,205],[185,145],[275,110],[655,85],[720,115],[755,175],[760,255],[720,325],[650,365]],
    "saga": [[350,355],[235,355],[150,325],[110,260],[110,185],[155,125],[235,95],[635,95],[715,125],[755,185],[760,265],[715,325],[640,355]],
}

DISTANCES = {
    "sapporo:turf": [1000,1200,1500,1800,2000,2600], "sapporo:dirt": [1000,1700,2400],
    "hakodate:turf": [1000,1200,1700,1800,2000,2600], "hakodate:dirt": [1000,1700,2400],
    "fukushima:turf": [1000,1200,1700,1800,2000,2600], "fukushima:dirt": [1000,1150,1700,2400],
    "niigata:inner": [1200,1400,2000,2200,2400], "niigata:outer": [1400,1600,1800,2000,3000,3200], "niigata:straight": [1000], "niigata:dirt": [1000,1200,1700,1800,2500],
    "tokyo:turf": [1400,1600,1800,2000,2300,2400,2500,2600,3400], "tokyo:dirt": [1200,1300,1400,1600,2100,2400],
    "nakayama:inner": [1800,2000,2500,3600], "nakayama:outer": [1200,1600,2200,2600,4000], "nakayama:dirt": [1000,1200,1700,1800,2400,2500],
    "chukyo:turf": [1200,1300,1400,1600,2000,2200,3000], "chukyo:dirt": [1200,1400,1800,1900,2500],
    "kyoto:inner": [1100,1200,1400,1600,2000], "kyoto:outer": [1400,1600,1800,2000,2200,2400,3000,3200], "kyoto:dirt": [1000,1100,1200,1400,1800,1900,2600],
    "hanshin:inner": [1200,1400,2000,2200,3000], "hanshin:outer": [1400,1600,1800,2400,2600], "hanshin:dirt": [1200,1400,1800,2000,2600],
    "kokura:turf": [1000,1200,1700,1800,2000,2600], "kokura:dirt": [1000,1700,2400],
    "monbetsu:outer": [1000,1100,1200,1700,1800,2000,2600], "monbetsu:inner": [1500,1600],
    "morioka:turf": [1000,1600,1700,2400], "morioka:dirt": [1000,1200,1400,1600,1800,2000,2500,2600],
    "mizusawa": [850,1300,1400,1600,1800,1900,2000], "urawa": [800,1300,1400,1500,1600,1900,2000],
    "funabashi:outer": [1000,1200,1600,1700,1800,2000,2200], "funabashi:inner": [1400,1500],
    "oi:outer_right": [1000,1200,1400,1700,1800,2000,2400,2600], "oi:inner_right": [1500,1600], "oi:outer_left": [1650],
    "kawasaki": [900,1400,1500,1600,2000,2100], "kanazawa": [900,1400,1500,1700,1900,2000,2100],
    "kasamatsu": [800,1400,1580,1800,1900,2500], "nagoya": [920,1500,1700,2000,2100],
    "sonoda": [820,1230,1400,1700,1870,2400], "himeji": [800,1400,1500,1800,2000],
    "kochi": [800,1300,1400,1600,1800,1900,2400], "saga": [900,1300,1400,1750,1800,2000,2500],
}


def chart_name(slug: str, surface: str, layout: str) -> str:
    if surface == "ダート": return "pic_course_durt.png"
    special = {
        "niigata": {"inner":"pic_course_shiba_in.png", "outer":"pic_course_shiba_out.png", "straight":"pic_course_shiba_straight.png"},
        "nakayama": {"inner":"pic_course_turf.png", "outer":"pic_course_turf2.png"},
        "kyoto": {"inner":"pic_course_shiba_in.png", "outer":"pic_course_shiba_out.png"},
        "hanshin": {"inner":"pic_course_turf.png", "outer":"pic_course_turf2.png"},
    }
    return special.get(slug, {}).get(layout, "pic_course_turf.png" if slug == "tokyo" else "pic_course_shiba.png")


def simplify(points: list[list[float]], tolerance: float) -> list[list[float]]:
    if len(points) <= 2: return points
    a, b = points[0], points[-1]
    errors = [abs(y - (a[1] + (b[1]-a[1])*(x-a[0])/(b[0]-a[0]))) for x,y in points[1:-1]]
    error = max(errors, default=0)
    if error <= tolerance: return [a,b]
    index = errors.index(error)+1
    return simplify(points[:index+1],tolerance)[:-1]+simplify(points[index:],tolerance)


def digitize(path: Path, lap: float, elevation: float | None, direction: str) -> tuple[list, dict]:
    image = Image.open(path).convert("RGB")
    colors = Counter(getattr(image, "get_flattened_data", image.getdata)())
    is_dirt = "durt" in path.name
    def match(c):
        r,g,b = c
        return (r > g*1.12 and g > b*1.12 and g > 70) if is_dirt else (g > r*1.12 and g > b*1.12 and r > 40)
    color = next(c for c,n in colors.most_common(80) if match(c))
    pixels = image.load()
    columns = {}
    bottom = 0
    for x in range(image.width):
        ys = [y for y in range(image.height) if pixels[x,y] == color]
        if len(ys) >= 4:
            columns[x] = min(ys); bottom = max(bottom,max(ys))
    x0,x1 = min(columns),max(columns)
    values = []
    for i in range(121):
        x = x0+(x1-x0)*i/120
        neighbors = [columns[k] for k in range(round(x)-3,round(x)+4) if k in columns]
        values.append(statistics.median(neighbors) if neighbors else columns[min(columns,key=lambda k:abs(k-x))])
    if direction in {"left", "straight"}: values.reverse()
    zero = values[0]
    scale = elevation/(max(values)-min(values)) if elevation is not None else 2/max(1,bottom-zero)
    points = [[round(lap*i/120,2),round((zero-y)*scale,3)] for i,y in enumerate(values)]
    points[0][1] = 0
    if direction != "straight": points[-1][1] = 0
    return simplify(points,max(0.025,(elevation or 1.2)*0.012)), {"pixel_bounds":[x0,x1],"fill_rgb":list(color),"method":"official_diagram_color_trace_scaled_to_published_range"}


def build(reference_dir: Path) -> dict:
    sources = json.loads((reference_dir/"jra_sources.json").read_text(encoding="utf-8"))
    images = {Path(i["file"]).name:i["url"] for s in sources for i in s["images"] if "file" in i}
    nar = json.loads((reference_dir/"nar_sources.json").read_text(encoding="utf-8"))
    nar_maps = {i["alt"].split("競馬場")[0]:i["url"] for i in nar}
    profiles = []
    for group, items in [("central",_COURSE_PROFILES),("local",_LOCAL_COURSE_PROFILES)]:
        for (course,surface,layout), native in items.items():
            slug=SLUGS[course];kind="turf" if surface=="芝" else "dirt";p=asdict(native)
            lap=p.get("lap_length_m") or 1000.0
            row={"id":f"{slug}-{kind}-{layout}","course":course,"surface":surface,"layout":layout,"label":f"{course}・{surface}・{LABELS[layout]}","group":group,"direction":p["direction"],"lap_m":p.get("lap_length_m"),"straight_m":p.get("straight_length_m",p.get("home_straight_m")),"elevation_m":p["elevation_m"],"source_url":p["source_url"],"checked_at":"2026-11-01","plan_accuracy":"official_plan_simplified_not_surveyed","start_accuracy":"lap_offset_approximation_chutes_omitted"}
            plan_key=f"{slug}:{kind}" if f"{slug}:{kind}" in PLANS else f"{slug}:{layout}"
            plan=PLANS.get(plan_key,PLANS.get(slug))
            if plan is None and slug=="nakayama":plan=PLANS["nakayama:inner"]
            if plan is None and slug=="hanshin":plan=PLANS["hanshin:inner"]
            # The left-running Oi finish is on the opposite side of its home straight.
            if slug=="oi" and p["direction"]=="left":plan=[[580,355]]+list(reversed(PLANS["oi"][1:]))
            if layout=="straight":plan=[[110,245],[750,245]]
            row["plan_points"]=plan
            dist_key=next(k for k in [f"{slug}:{layout}",f"{slug}:{kind}",slug] if k in DISTANCES)
            row["distances_m"]=DISTANCES[dist_key]
            if group=="central":
                name=f"{slug}_{chart_name(slug,surface,layout)}"
                row["elevation_points"],trace=digitize(reference_dir/name,lap,p["elevation_m"],p["direction"])
                row["elevation_accuracy"]="diagram_approximation"
                row["elevation_source_url"]=images[name]
                row["plan_source_url"]=images[f"{slug}_pic_course_heimenzu.png"]
                row["trace"]=trace
            else:
                row["plan_source_url"]=nar_maps[course]
                row["elevation_source_url"]=row["source_url"]
                row["elevation_points"]=[[0,0],[lap,0]] if p["elevation_m"]==0 else None
                row["elevation_accuracy"]="official_flat" if p["elevation_m"]==0 else "range_only"
                if course=="盛岡":
                    forward=([[0,0],[100,0],[500,.8],[850,2.6],[900,2.5],[1100,.7],[1300,-1.4],[1350,-1.8],[1550,0],[1600,0]] if kind=="dirt" else [[0,0],[100,0],[250,-.5],[300,-.5],[450,0],[600,.7],[750,1.4],[800,1.3],[1000,-1.1],[1130,-3.1],[1170,-3.2],[1300,-.7],[1370,0],[1400,0]])
                    row["elevation_points"]=[[lap-x,y] for x,y in reversed(forward)]
                    row["elevation_accuracy"]="diagram_approximation"
                    row["elevation_source_url"]="https://www.iwatekeiba.or.jp/dir/wp-content/uploads/2017/02/oro_ud_766.jpg"
                    row["source_url"]="https://www.iwatekeiba.or.jp/oro_01"
            profiles.append(row)
    return {"schema_version":1,"checked_at":"2026-11-01","profiles":profiles}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references",type=Path,default=ROOT/"analysis_outputs/simulator_course_references")
    parser.add_argument("--output",type=Path,default=ROOT/"frontend/src/fixtures/raceCourses.json")
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to((ROOT/"frontend/src/fixtures").resolve()):parser.error("Output must remain in frontend/src/fixtures")
    payload=build(args.references)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(payload['profiles'])} course profiles in {len({p['course'] for p in payload['profiles']})} venues")


if __name__=="__main__":main()
