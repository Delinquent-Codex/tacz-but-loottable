# Spawn a Rare Warzone supply chest at the executor's location
setblock ~ ~ ~ chest[facing=north]
data modify block ~ ~ ~ LootTable set value "tacz:warzone/chest_rare"
data modify block ~ ~ ~ CustomName set value '{"text":"Warzone Supply (Rare)","color":"aqua","italic":false}'
