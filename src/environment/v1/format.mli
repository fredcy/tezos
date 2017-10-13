(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Pierre Weis, projet Cristal, INRIA Rocquencourt            *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove channel functions
   * Remove toplevel effect based functions
   * Remove deprecated functions

*)

(** Pretty printing.

   This module implements a pretty-printing facility to format values
   within 'pretty-printing boxes'. The pretty-printer splits lines
   at specified break hints, and indents lines according to the box
   structure.

   For a gentle introduction to the basics of pretty-printing using
   [Format], read
   {{:http://caml.inria.fr/resources/doc/guides/format.en.html}
    http://caml.inria.fr/resources/doc/guides/format.en.html}.

   You may consider this module as providing an extension to the
   [printf] facility to provide automatic line splitting. The addition of
   pretty-printing annotations to your regular [printf] formats gives you
   fancy indentation and line breaks.
   Pretty-printing annotations are described below in the documentation of
   the function {!Format.fprintf}.

   You may also use the explicit box management and printing functions
   provided by this module. This style is more basic but more verbose
   than the [fprintf] concise formats.

   For instance, the sequence
   [open_box 0; print_string "x ="; print_space ();
    print_int 1; close_box (); print_newline ()]
   that prints [x = 1] within a pretty-printing box, can be
   abbreviated as [printf "@[%s@ %i@]@." "x =" 1], or even shorter
   [printf "@[x =@ %i@]@." 1].

   Rule of thumb for casual users of this library:
 - use simple boxes (as obtained by [open_box 0]);
 - use simple break hints (as obtained by [print_cut ()] that outputs a
   simple break hint, or by [print_space ()] that outputs a space
   indicating a break hint);
 - once a box is opened, display its material with basic printing
   functions (e. g. [print_int] and [print_string]);
 - when the material for a box has been printed, call [close_box ()] to
   close the box;
 - at the end of your routine, flush the pretty-printer to display all the
   remaining material, e.g. evaluate [print_newline ()].

   The behaviour of pretty-printing commands is unspecified
   if there is no opened pretty-printing box. Each box opened via
   one of the [open_] functions below must be closed using [close_box]
   for proper formatting. Otherwise, some of the material printed in the
   boxes may not be output, or may be formatted incorrectly.

   In case of interactive use, the system closes all opened boxes and
   flushes all pending text (as with the [print_newline] function)
   after each phrase. Each phrase is therefore executed in the initial
   state of the pretty-printer.

   Warning: the material output by the following functions is delayed
   in the pretty-printer queue in order to compute the proper line
   splitting. Hence, you should not mix calls to the printing functions
   of the basic I/O system with calls to the functions of this module:
   this could result in some strange output seemingly unrelated with
   the evaluation order of printing commands.
*)

(** {6:tags Semantic Tags} *)

type tag = string

(** {6:meaning Changing the meaning of standard formatter pretty printing} *)

(** The [Format] module is versatile enough to let you completely redefine
  the meaning of pretty printing: you may provide your own functions to define
  how to handle indentation, line splitting, and even printing of all the
  characters that have to be printed! *)

type formatter_out_functions = {
  out_string : string -> int -> int -> unit;
  out_flush : unit -> unit;
  out_newline : unit -> unit;
  out_spaces : int -> unit;
}

(** {6:tagsmeaning Changing the meaning of printing semantic tags} *)

type formatter_tag_functions = {
  mark_open_tag : tag -> string;
  mark_close_tag : tag -> string;
  print_open_tag : tag -> unit;
  print_close_tag : tag -> unit;
}
(** The tag handling functions specific to a formatter:
  [mark] versions are the 'tag marking' functions that associate a string
  marker to a tag in order for the pretty-printing engine to flush
  those markers as 0 length tokens in the output device of the formatter.
  [print] versions are the 'tag printing' functions that can perform
  regular printing when a tag is closed or opened. *)

(** {6 Multiple formatted output} *)

type formatter
(** Abstract data corresponding to a pretty-printer (also called a
  formatter) and all its machinery.

  Defining new pretty-printers permits unrelated output of material in
  parallel on several output channels.
  All the parameters of a pretty-printer are local to a formatter:
  margin, maximum indentation limit, maximum number of boxes
  simultaneously opened, ellipsis, and so on, are specific to
  each pretty-printer and may be fixed independently.
  Given a [Pervasives.out_channel] output channel [oc], a new formatter
  writing to that channel is simply obtained by calling
  [formatter_of_out_channel oc].
  Alternatively, the [make_formatter] function allocates a new
  formatter with explicit output and flushing functions
  (convenient to output material to strings for instance).
*)

val formatter_of_buffer : Buffer.t -> formatter
(** [formatter_of_buffer b] returns a new formatter writing to
  buffer [b]. As usual, the formatter has to be flushed at
  the end of pretty printing, using [pp_print_flush] or
  [pp_print_newline], to display all the pending material. *)

val make_formatter :
  (string -> int -> int -> unit) -> (unit -> unit) -> formatter
(** [make_formatter out flush] returns a new formatter that writes according
  to the output function [out], and the flushing function [flush]. For
  instance, a formatter to the [Pervasives.out_channel] [oc] is returned by
  [make_formatter (Pervasives.output oc) (fun () -> Pervasives.flush oc)]. *)

(** {6 Basic functions to use with formatters} *)

val pp_open_hbox : formatter -> unit -> unit
val pp_open_vbox : formatter -> int -> unit
val pp_open_hvbox : formatter -> int -> unit
val pp_open_hovbox : formatter -> int -> unit
val pp_open_box : formatter -> int -> unit
val pp_close_box : formatter -> unit -> unit
val pp_open_tag : formatter -> string -> unit
val pp_close_tag : formatter -> unit -> unit
val pp_print_string : formatter -> string -> unit
val pp_print_as : formatter -> int -> string -> unit
val pp_print_int : formatter -> int -> unit
val pp_print_float : formatter -> float -> unit
val pp_print_char : formatter -> char -> unit
val pp_print_bool : formatter -> bool -> unit
val pp_print_break : formatter -> int -> int -> unit
val pp_print_cut : formatter -> unit -> unit
val pp_print_space : formatter -> unit -> unit
val pp_force_newline : formatter -> unit -> unit
val pp_print_flush : formatter -> unit -> unit
val pp_print_newline : formatter -> unit -> unit
val pp_print_if_newline : formatter -> unit -> unit
val pp_set_tags : formatter -> bool -> unit
val pp_set_print_tags : formatter -> bool -> unit
val pp_set_mark_tags : formatter -> bool -> unit
val pp_get_print_tags : formatter -> unit -> bool
val pp_get_mark_tags : formatter -> unit -> bool
val pp_set_margin : formatter -> int -> unit
val pp_get_margin : formatter -> unit -> int
val pp_set_max_indent : formatter -> int -> unit
val pp_get_max_indent : formatter -> unit -> int
val pp_set_max_boxes : formatter -> int -> unit
val pp_get_max_boxes : formatter -> unit -> int
val pp_over_max_boxes : formatter -> unit -> bool
val pp_set_ellipsis_text : formatter -> string -> unit
val pp_get_ellipsis_text : formatter -> unit -> string

val pp_set_formatter_output_functions :
  formatter -> (string -> int -> int -> unit) -> (unit -> unit) -> unit

val pp_get_formatter_output_functions :
  formatter -> unit -> (string -> int -> int -> unit) * (unit -> unit)

val pp_set_formatter_tag_functions :
  formatter -> formatter_tag_functions -> unit

val pp_get_formatter_tag_functions :
  formatter -> unit -> formatter_tag_functions

val pp_set_formatter_out_functions :
  formatter -> formatter_out_functions -> unit

val pp_get_formatter_out_functions :
  formatter -> unit -> formatter_out_functions
(** These functions are the basic ones: usual functions
   operating on the standard formatter are defined via partial
   evaluation of these primitives. For instance,
   [print_string] is equal to [pp_print_string std_formatter]. *)

val pp_flush_formatter : formatter -> unit
(** [pp_flush_formatter fmt] flushes [fmt]'s internal queue, ensuring that all
    the printing and flushing actions have been performed. In addition, this
    operation will close all boxes and reset the state of the formatter.

    This will not flush [fmt]'s output. In most cases, the user may want to use
    {!pp_print_flush} instead. *)

(** {6 Convenience formatting functions.} *)

val pp_print_list:
  ?pp_sep:(formatter -> unit -> unit) ->
  (formatter -> 'a -> unit) -> (formatter -> 'a list -> unit)
(** [pp_print_list ?pp_sep pp_v ppf l] prints items of list [l],
  using [pp_v] to print each item, and calling [pp_sep]
  between items ([pp_sep] defaults to {!pp_print_cut}).
  Does nothing on empty lists.

  @since 4.02.0
*)

val pp_print_text : formatter -> string -> unit
(** [pp_print_text ppf s] prints [s] with spaces and newlines
  respectively printed with {!pp_print_space} and
  {!pp_force_newline}.

  @since 4.02.0
*)

(** {6 [printf] like functions for pretty-printing.} *)

val fprintf : formatter -> ('a, formatter, unit) format -> 'a

(** [fprintf ff fmt arg1 ... argN] formats the arguments [arg1] to [argN]
  according to the format string [fmt], and outputs the resulting string on
  the formatter [ff].

  The format [fmt] is a character string which contains three types of
  objects: plain characters and conversion specifications as specified in
  the [Printf] module, and pretty-printing indications specific to the
  [Format] module.

  The pretty-printing indication characters are introduced by
  a [@] character, and their meanings are:
  - [@\[]: open a pretty-printing box. The type and offset of the
    box may be optionally specified with the following syntax:
    the [<] character, followed by an optional box type indication,
    then an optional integer offset, and the closing [>] character.
    Box type is one of [h], [v], [hv], [b], or [hov].
    '[h]' stands for an 'horizontal' box,
    '[v]' stands for a 'vertical' box,
    '[hv]' stands for an 'horizontal-vertical' box,
    '[b]' stands for an 'horizontal-or-vertical' box demonstrating indentation,
    '[hov]' stands a simple 'horizontal-or-vertical' box.
    For instance, [@\[<hov 2>] opens an 'horizontal-or-vertical'
    box with indentation 2 as obtained with [open_hovbox 2].
    For more details about boxes, see the various box opening
    functions [open_*box].
  - [@\]]: close the most recently opened pretty-printing box.
  - [@,]: output a 'cut' break hint, as with [print_cut ()].
  - [@ ]: output a 'space' break hint, as with [print_space ()].
  - [@;]: output a 'full' break hint as with [print_break]. The
    [nspaces] and [offset] parameters of the break hint may be
    optionally specified with the following syntax:
    the [<] character, followed by an integer [nspaces] value,
    then an integer [offset], and a closing [>] character.
    If no parameters are provided, the good break defaults to a
    'space' break hint.
  - [@.]: flush the pretty printer and split the line, as with
    [print_newline ()].
  - [@<n>]: print the following item as if it were of length [n].
    Hence, [printf "@<0>%s" arg] prints [arg] as a zero length string.
    If [@<n>] is not followed by a conversion specification,
    then the following character of the format is printed as if
    it were of length [n].
  - [@\{]: open a tag. The name of the tag may be optionally
    specified with the following syntax:
    the [<] character, followed by an optional string
    specification, and the closing [>] character. The string
    specification is any character string that does not contain the
    closing character ['>']. If omitted, the tag name defaults to the
    empty string.
    For more details about tags, see the functions [open_tag] and
    [close_tag].
  - [@\}]: close the most recently opened tag.
  - [@?]: flush the pretty printer as with [print_flush ()].
    This is equivalent to the conversion [%!].
  - [@\n]: force a newline, as with [force_newline ()], not the normal way
    of pretty-printing, you should prefer using break hints inside a vertical
    box.

  Note: If you need to prevent the interpretation of a [@] character as a
  pretty-printing indication, you must escape it with a [%] character.
  Old quotation mode [@@] is deprecated since it is not compatible with
  formatted input interpretation of character ['@'].

  Example: [printf "@[%s@ %d@]@." "x =" 1] is equivalent to
  [open_box (); print_string "x ="; print_space ();
   print_int 1; close_box (); print_newline ()].
  It prints [x = 1] within a pretty-printing 'horizontal-or-vertical' box.

*)

val sprintf : ('a, unit, string) format -> 'a
(** Same as [printf] above, but instead of printing on a formatter,
  returns a string containing the result of formatting the arguments.
  Note that the pretty-printer queue is flushed at the end of {e each
  call} to [sprintf].

  In case of multiple and related calls to [sprintf] to output
  material on a single string, you should consider using [fprintf]
  with the predefined formatter [str_formatter] and call
  [flush_str_formatter ()] to get the final result.

  Alternatively, you can use [Format.fprintf] with a formatter writing to a
  buffer of your own: flushing the formatter and the buffer at the end of
  pretty-printing returns the desired string.
*)

val asprintf : ('a, formatter, unit, string) format4 -> 'a
(** Same as [printf] above, but instead of printing on a formatter,
  returns a string containing the result of formatting the arguments.
  The type of [asprintf] is general enough to interact nicely with [%a]
  conversions.
  @since 4.01.0
*)

val ifprintf : formatter -> ('a, formatter, unit) format -> 'a
(** Same as [fprintf] above, but does not print anything.
  Useful to ignore some material when conditionally printing.
  @since 3.10.0
*)

(** Formatted output functions with continuations. *)

val kfprintf :
  (formatter -> 'a) -> formatter ->
  ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [fprintf] above, but instead of returning immediately,
  passes the formatter to its first argument at the end of printing. *)

val ikfprintf :
  (formatter -> 'a) -> formatter ->
  ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [kfprintf] above, but does not print anything.
  Useful to ignore some material when conditionally printing.
  @since 3.12.0
*)

val ksprintf : (string -> 'a) -> ('b, unit, string, 'a) format4 -> 'b
(** Same as [sprintf] above, but instead of returning the string,
  passes it to the first argument. *)

val kasprintf : (string -> 'a) -> ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [asprintf] above, but instead of returning the string,
  passes it to the first argument.
  @since 4.03
*)
