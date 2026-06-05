import json, glob, os
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
def dig_tokens(obj):
    # find any total_tokens / input+output tokens in nested dicts
    found={}
    def rec(o,path=""):
        if isinstance(o,dict):
            for k,v in o.items():
                kl=k.lower()
                if isinstance(v,(int,float)) and ("token" in kl):
                    found[path+"/"+k]=v
                rec(v,path+"/"+k)
        elif isinstance(o,list):
            for i,v in enumerate(o):
                rec(v,path+f"[{i}]")
    rec(obj)
    return found
out=[]
for c in cells:
    rj = glob.glob(os.path.join(base,c,"*","*","result.json"))
    rec={"cell":c}
    if not rj:
        rec["err"]="no result.json"; out.append(rec); continue
    r=json.load(open(rj[0]))
    rec["error"]=r.get("error")
    rec["error_category"]=r.get("error_category")
    rec["partial_trajectory"]=r.get("partial_trajectory")
    rec["trajectory_source"]=r.get("trajectory_source")
    rec["model"]=r.get("model")
    rec["skill_mode"]=r.get("skill_mode")
    rec["include_task_skills"]=r.get("include_task_skills")
    rec["n_tool_calls"]=r.get("n_tool_calls")
    rec["usage_tracking"]=r.get("usage_tracking")
    rec["rewards_field"]=r.get("rewards")
    fm=r.get("final_metrics")
    rec["final_metrics"]=fm
    # token dig
    toks=dig_tokens(r)
    rec["token_fields"]=toks
    out.append(rec)
print(json.dumps(out,indent=1,default=str))
