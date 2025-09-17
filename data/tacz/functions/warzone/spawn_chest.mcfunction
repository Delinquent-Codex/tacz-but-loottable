# Spawn a Warzone supply chest at the executor's location with randomized loot
setblock ~ ~ ~ chest[facing=north]
data modify block ~ ~ ~ LootTable set value "tacz:warzone/chest"
data modify block ~ ~ ~ CustomName set value '{"text":"Warzone Supply (Mixed)","color":"yellow","italic":false}'
