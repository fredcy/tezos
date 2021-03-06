(**************************************************************************)
(*                                                                        *)
(*    Copyright (c) 2014 - 2016.                                          *)
(*    Dynamic Ledger Solutions, Inc. <contact@tezos.com>                  *)
(*                                                                        *)
(*    All rights reserved. No warranty, explicit or implicit, provided.   *)
(*                                                                        *)
(**************************************************************************)

val tez_sym: string

open Cli_entries
val init_arg: (string, Client_commands.context) arg
val fee_arg: (Tez.t, Client_commands.context) arg
val arg_arg: (string, Client_commands.context) arg
val source_arg: (string option, Client_commands.context) arg

val delegate_arg: (string option, Client_commands.context) arg
val delegatable_switch: (bool, Client_commands.context) arg
val non_spendable_switch: (bool, Client_commands.context) arg
val max_priority_arg: (int option, Client_commands.context) arg
val free_mining_switch: (bool, Client_commands.context) arg
val force_switch: (bool, Client_commands.context) arg
val endorsement_delay_arg: (int, Client_commands.context) arg

val tez_arg :
  default:string ->
  parameter:string ->
  doc:string ->
  (Tez.t, Client_commands.context) arg
val tez_param :
  name:string ->
  desc:string ->
  ('a, Client_commands.context, 'ret) Cli_entries.params ->
  (Tez.t -> 'a, Client_commands.context, 'ret) Cli_entries.params

module Daemon : sig
  val mining_switch: (bool, Client_commands.context) arg
  val endorsement_switch: (bool, Client_commands.context) arg
  val denunciation_switch: (bool, Client_commands.context) arg
end
