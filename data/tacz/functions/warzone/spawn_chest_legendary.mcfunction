# Spawn a Legendary Warzone supply chest at the executor's location
setblock ~ ~ ~ chest[facing=north]
data modify block ~ ~ ~ LootTable set value "tacz:warzone/chest_legendary"
data modify block ~ ~ ~ CustomName set value '{"text":"Warzone Supply (Legendary)","color":"gold","italic":false}'
