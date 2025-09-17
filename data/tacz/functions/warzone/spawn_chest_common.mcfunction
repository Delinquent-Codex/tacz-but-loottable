# Spawn a Common Warzone supply chest at the executor's location
setblock ~ ~ ~ chest[facing=north]
data modify block ~ ~ ~ LootTable set value "tacz:warzone/chest_common"
data modify block ~ ~ ~ CustomName set value '{"text":"Warzone Supply (Common)","color":"gray","italic":false}'
