from txt2img import txt2img

prompts = {
    "hada": """old man <lora:KyoAni:1.5>""", # DONE
    "ia3": """old man <lora:Cammy-400:1>""", # DONE
    "lokr": """old man <lora:mikapikazo:1>""",  # DONE
    "full": """old man <lora:test_full:1>""", #DONE
    "lora": """old man <lora:animeScreencapStyle_v230epochs:1>""", # DONE

}

i = 0
for type, prompt in prompts.items():    
    image = txt2img(
        checkpoint = "juggernaut_final",
        positive = prompt,
        negative = """""",
        steps = 10,
        seed = 3125748766
    )

    image.save("../output/" + type + ".png")
    i += 1
