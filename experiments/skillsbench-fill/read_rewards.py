import json, glob, os, sys
cells = [
 "opus-4.8__with__glm-lake-mendota__t3",
 "opus-4.8__with__gravitational-wave-detection__t1",
 "opus-4.8__with__gravitational-wave-detection__t3",
 "opus-4.8__with__grid-dispatch-operator__t1",
 "opus-4.8__with__grid-dispatch-operator__t2",
 "opus-4.8__with__grid-dispatch-operator__t3",
 "opus-4.8__with__hvac-control__t1",
 "opus-4.8__with__hvac-control__t2",
 "opus-4.8__with__hvac-control__t3",
 "opus-4.8__with__invoice-fraud-detection__t1",
 "opus-4.8__with__invoice-fraud-detection__t2",
 "opus-4.8__with__invoice-fraud-detection__t3",
]
base="/home/bingran_you/sb-fill/jobs"
out=[]
for c in cells:
    rj = glob.glob(os.path.join(base,c,"*","*","result.json"))
    rec={"cell":c}
    if not rj:
        rec["err"]="no result.json"; out.append(rec); continue
    rdir=os.path.dirname(rj[0])
    try:
        r=json.load(open(rj[0]))
        rec["result_reward"]=r.get("reward")
        rec["result_keys"]=list(r.keys())
    except Exception as e:
        rec["result_err"]=str(e)
    # rewards.jsonl
    rl=os.path.join(rdir,"rewards.jsonl")
    if os.path.exists(rl):
        lines=[l for l in open(rl).read().splitlines() if l.strip()]
        rec["rewards_jsonl"]=lines
    # ctrf
    ctrf=os.path.join(rdir,"verifier","ctrf.json")
    if os.path.exists(ctrf):
        try:
            cc=json.load(open(ctrf))
            summ=cc.get("results",{}).get("summary",{})
            rec["ctrf_summary"]=summ
        except Exception as e:
            rec["ctrf_err"]=str(e)
    # tokens: look in trajectory / agent
    out.append(rec)
print(json.dumps(out,indent=1))
