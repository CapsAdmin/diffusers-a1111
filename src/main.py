from txt2img import txt2img

prompts = {
    "hada": """old man <lora:KyoAni:1.5>""", 
    "ia3": """old man <lora:Cammy-400:1>""", # DONE
    "lokr": """old man <lora:mikapikazo:1>""",  # DONE
    "full": """old man <lora:test_full:1>""",
    "lora": """old man <lora:animeScreencapStyle_v230epochs:1>""", # DONE

}

image = txt2img(
    checkpoint = "juggernaut_final",
    positive = prompts["lokr"],
    negative = """""",
    steps = 10,
    seed = 3125748766
)

image.save("output.png")