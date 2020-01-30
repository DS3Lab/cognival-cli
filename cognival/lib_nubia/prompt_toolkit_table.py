# Source: https://github.com/prompt-toolkit/python-prompt-toolkit/blob/91ebdd3bf410f16a0ffb11d306e9f3d6f5902bc7/prompt_toolkit/widgets/table.py
# Source: https://github.com/prompt-toolkit/python-prompt-toolkit/blob/master/examples/full-screen/simple-demos/focus.py
#!/usr/bin/env python3
import textwrap

import numpy as np

from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import Dimension as D, sum_layout_dimensions, max_layout_dimensions, to_dimension
from prompt_toolkit.widgets import Box, TextArea, Label, Button, HorizontalLine
# from prompt_toolkit.widgets.base import Border
from prompt_toolkit.layout.containers import Window, VSplit, HSplit, HorizontalAlign, VerticalAlign
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.utils import take_using_weights


class SpaceBorder:
    " Box drawing characters. (Spaces) "
    HORIZONTAL = ' '
    VERTICAL = ' '

    TOP_LEFT = ' '
    TOP_RIGHT = ' '
    BOTTOM_LEFT = ' '
    BOTTOM_RIGHT = ' '

    LEFT_T = ' '
    RIGHT_T = ' '
    TOP_T = ' '
    BOTTOM_T = ' '

    INTERSECT = ' '


class AsciiBorder:
    " Box drawing characters. (ASCII) "
    HORIZONTAL = '-'
    VERTICAL = '|'

    TOP_LEFT = '+'
    TOP_RIGHT = '+'
    BOTTOM_LEFT = '+'
    BOTTOM_RIGHT = '+'

    LEFT_T = '+'
    RIGHT_T = '+'
    TOP_T = '+'
    BOTTOM_T = '+'

    INTERSECT = '+'


class ThinBorder:
    " Box drawing characters. (Thin) "
    HORIZONTAL = '\u2500'
    VERTICAL = '\u2502'

    TOP_LEFT = '\u250c'
    TOP_RIGHT = '\u2510'
    BOTTOM_LEFT = '\u2514'
    BOTTOM_RIGHT = '\u2518'

    LEFT_T = '\u251c'
    RIGHT_T = '\u2524'
    TOP_T = '\u252c'
    BOTTOM_T = '\u2534'

    INTERSECT = '\u253c'


class RoundedBorder(ThinBorder):
    " Box drawing characters. (Rounded) "
    TOP_LEFT = '\u256d'
    TOP_RIGHT = '\u256e'
    BOTTOM_LEFT = '\u2570'
    BOTTOM_RIGHT = '\u256f'


class ThickBorder:
    " Box drawing characters. (Thick) "
    HORIZONTAL = '\u2501'
    VERTICAL = '\u2503'

    TOP_LEFT = '\u250f'
    TOP_RIGHT = '\u2513'
    BOTTOM_LEFT = '\u2517'
    BOTTOM_RIGHT = '\u251b'

    LEFT_T = '\u2523'
    RIGHT_T = '\u252b'
    TOP_T = '\u2533'
    BOTTOM_T = '\u253b'

    INTERSECT = '\u254b'


class DoubleBorder:
    " Box drawing characters. (Thin) "
    HORIZONTAL = '\u2550'
    VERTICAL = '\u2551'

    TOP_LEFT = '\u2554'
    TOP_RIGHT = '\u2557'
    BOTTOM_LEFT = '\u255a'
    BOTTOM_RIGHT = '\u255d'

    LEFT_T = '\u2560'
    RIGHT_T = '\u2563'
    TOP_T = '\u2566'
    BOTTOM_T = '\u2569'

    INTERSECT = '\u256c'


class Merge:
    def __init__(self, cell, merge=1):
        self.cell = cell
        self.merge = merge

    def __iter__(self):
        yield self.cell
        yield self.merge


class Table(HSplit):
    def __init__(self, table,
                 borders=ThinBorder, column_width=None, column_widths=[],
                 window_too_small=None, align=VerticalAlign.JUSTIFY,
                 padding=0, padding_char=None, padding_style='',
                 width=None, height=None, z_index=None,
                 modal=False, key_bindings=None, style=''):
        self.borders = borders
        self.column_width = column_width
        self.column_widths = column_widths

        # ensure the table is iterable (has rows)
        if not isinstance(table, list):
            table = [table]
        children = [_Row(row=row, table=self, borders=borders, height=self.get_row_height(row))
                    for row in table]

        super().__init__(
            children=children,
            window_too_small=window_too_small,
            align=align,
            padding=padding,
            padding_char=padding_char,
            padding_style=padding_style,
            width=width,
            height=height,
            z_index=z_index,
            modal=modal,
            key_bindings=key_bindings,
            style=style)

    def get_row_height(self, row):
        if isinstance(row, Merge):
            return None
        column_widths_min = [cw.min for cw in self.column_widths]
        height = 1
        for col, col_width in zip(row, column_widths_min):
            if isinstance(col, Label):
                wrapped_list = textwrap.wrap(col.text, width=col_width)
                col.text = "\n".join(wrapped_list)
                new_height = len(wrapped_list)
                if new_height > height:
                    height = new_height

        return min(4, height)

    @property
    def columns(self):
        return max(row.raw_columns for row in self.children)

    @property
    def _all_children(self):
        """
        List of child objects, including padding & borders.
        """
        def get():
            result = []

            # Padding top.
            if self.align in (VerticalAlign.CENTER, VerticalAlign.BOTTOM):
                result.append(Window(width=D(preferred=0)))

            # Border top is first inserted in children loop.

            # The children with padding.
            prev = None
            for child in self.children:
                result.append(_Border(
                    prev=prev,
                    next=child,
                    table=self,
                    borders=self.borders))
                result.append(child)
                prev = child

            # Border bottom.
            result.append(_Border(prev=prev, next=None, table=self, borders=self.borders))

            # Padding bottom.
            if self.align in (VerticalAlign.CENTER, VerticalAlign.TOP):
                result.append(Window(width=D(preferred=0)))

            return result

        return self._children_cache.get(tuple(self.children), get)

    def preferred_dimensions(self, width):
        dimensions = [[]] * self.columns
        for row in self.children:
            assert isinstance(row, _Row)
            j = 0
            for cell in row.children:
                assert isinstance(cell, _Cell)

                if cell.merge != 1:
                    dimensions[j].append(cell.preferred_width(width))

                j += cell.merge

        for i, c in enumerate(dimensions):
            yield D.exact(1)

            try:
                w = self.column_widths[i]
            except IndexError:
                w = self.column_width
            if w is None:  # fitted
                yield max_layout_dimensions(c)
            else:  # fixed or weighted
                yield to_dimension(w)
        yield D.exact(1)


class _VerticalBorder(Window):
    def __init__(self, borders):
        super().__init__(width=1, char=borders.VERTICAL)


class _HorizontalBorder(Window):
    def __init__(self, borders):
        super().__init__(height=1, char=borders.HORIZONTAL)


class _UnitBorder(Window):
    def __init__(self, char):
        super().__init__(width=1, height=1, char=char)


class _BaseRow(VSplit):
    @property
    def columns(self):
        return self.table.columns

    def _divide_widths(self, width):
        """
        Return the widths for all columns.
        Or None when there is not enough space.
        """
        children = self._all_children

        if not children:
            return []

        # Calculate widths.
        dimensions = list(self.table.preferred_dimensions(width))
        preferred_dimensions = [d.preferred for d in dimensions]

        # Sum dimensions
        sum_dimensions = sum_layout_dimensions(dimensions)

        # If there is not enough space for both.
        # Don't do anything.
        if sum_dimensions.min > width:
            return

        # Find optimal sizes. (Start with minimal size, increase until we cover
        # the whole width.)
        sizes = [d.min for d in dimensions]

        child_generator = take_using_weights(
            items=list(range(len(dimensions))),
            weights=[d.weight for d in dimensions])

        i = next(child_generator)

        # Increase until we meet at least the 'preferred' size.
        preferred_stop = min(width, sum_dimensions.preferred)

        while sum(sizes) < preferred_stop:
            if sizes[i] < preferred_dimensions[i]:
                sizes[i] += 1
            i = next(child_generator)

        # Increase until we use all the available space.
        max_dimensions = [d.max for d in dimensions]
        max_stop = min(width, sum_dimensions.max)

        while sum(sizes) < max_stop:
            if sizes[i] < max_dimensions[i]:
                sizes[i] += 1
            i = next(child_generator)

        # perform merges if necessary
        if len(children) != len(sizes):
            tmp = []
            i = 0
            for c in children:
                if isinstance(c, _Cell):
                    inc = (c.merge * 2) - 1
                    tmp.append(sum(sizes[i:i + inc]))
                else:
                    inc = 1
                    tmp.append(sizes[i])
                i += inc
            sizes = tmp

        return sizes


class _Row(_BaseRow):
    def __init__(self, row, table, borders,
                 window_too_small=None, align=HorizontalAlign.JUSTIFY,
                 padding=D.exact(0), padding_char=None, padding_style='',
                 width=None, height=None, z_index=None,
                 modal=False, key_bindings=None, style=''):
        self.table = table
        self.borders = borders

        # ensure the row is iterable (has cells)
        if not isinstance(row, list):
            row = [row]
        children = []
        for c in row:
            m = 1
            if isinstance(c, Merge):
                c, m = c
            elif isinstance(c, dict):
                c, m = Merge(**c)
            children.append(_Cell(cell=c, table=table, row=self, merge=m))

        super().__init__(
            children=children,
            window_too_small=window_too_small,
            align=align,
            padding=padding,
            padding_char=padding_char,
            padding_style=padding_style,
            width=width,
            height=height,
            z_index=z_index,
            modal=modal,
            key_bindings=key_bindings,
            style=style)

    @property
    def raw_columns(self):
        return sum(cell.merge for cell in self.children)

    @property
    def _all_children(self):
        """
        List of child objects, including padding & borders.
        """
        def get():
            result = []

            # Padding left.
            if self.align in (HorizontalAlign.CENTER, HorizontalAlign.RIGHT):
                result.append(Window(width=D(preferred=0)))

            # Border left is first inserted in children loop.

            # The children with padding.
            c = 0
            for child in self.children:
                result.append(_VerticalBorder(borders=self.borders))
                result.append(child)
                c += child.merge
            # Fill in any missing columns
            for _ in range(self.columns - c):
                result.append(_VerticalBorder(borders=self.borders))
                result.append(_Cell(cell=None, table=self.table, row=self))

            # Border right.
            result.append(_VerticalBorder(borders=self.borders))

            # Padding right.
            if self.align in (HorizontalAlign.CENTER, HorizontalAlign.LEFT):
                result.append(Window(width=D(preferred=0)))

            return result

        return self._children_cache.get(tuple(self.children), get)


class _Border(_BaseRow):
    def __init__(self, prev, next, table, borders,
                 window_too_small=None, align=HorizontalAlign.JUSTIFY,
                 padding=D.exact(0), padding_char=None, padding_style='',
                 width=None, height=None, z_index=None,
                 modal=False, key_bindings=None, style=''):
        assert prev or next
        self.prev = prev
        self.next = next
        self.table = table
        self.borders = borders

        children = [_HorizontalBorder(borders=borders)] * self.columns

        super().__init__(
            children=children,
            window_too_small=window_too_small,
            align=align,
            padding=padding,
            padding_char=padding_char,
            padding_style=padding_style,
            width=width,
            height=height or 1,
            z_index=z_index,
            modal=modal,
            key_bindings=key_bindings,
            style=style)

    def has_borders(self, row):
        yield None  # first (outer) border

        if not row:
            # this row is undefined, none of the borders need to be marked
            yield from [False] * (self.columns - 1)
        else:
            c = 0
            for child in row.children:
                yield from [False] * (child.merge - 1)
                yield True
                c += child.merge

            yield from [True] * (self.columns - c)

        yield None  # last (outer) border

    @property
    def _all_children(self):
        """
        List of child objects, including padding & borders.
        """
        def get():
            result = []

            # Padding left.
            if self.align in (HorizontalAlign.CENTER, HorizontalAlign.RIGHT):
                result.append(Window(width=D(preferred=0)))

            def char(i, pc=False, nc=False):
                if i == 0:
                    if self.prev and self.next:
                        return self.borders.LEFT_T
                    elif self.prev:
                        return self.borders.BOTTOM_LEFT
                    else:
                        return self.borders.TOP_LEFT

                if i == self.columns:
                    if self.prev and self.next:
                        return self.borders.RIGHT_T
                    elif self.prev:
                        return self.borders.BOTTOM_RIGHT
                    else:
                        return self.borders.TOP_RIGHT

                if pc and nc:
                    return self.borders.INTERSECT
                elif pc:
                    return self.borders.BOTTOM_T
                elif nc:
                    return self.borders.TOP_T
                else:
                    return self.borders.HORIZONTAL

            # Border left is first inserted in children loop.

            # The children with padding.
            pcs = self.has_borders(self.prev)
            ncs = self.has_borders(self.next)
            for i, (child, pc, nc) in enumerate(zip(self.children, pcs, ncs)):
                result.append(_UnitBorder(char=char(i, pc, nc)))
                result.append(child)

            # Border right.
            result.append(_UnitBorder(char=char(self.columns)))

            # Padding right.
            if self.align in (HorizontalAlign.CENTER, HorizontalAlign.LEFT):
                result.append(Window(width=D(preferred=0)))

            return result

        return self._children_cache.get(tuple(self.children), get)


class _Cell(HSplit):
    def __init__(self, cell, table, row, merge=1,
                 padding=0, char=None,
                 padding_left=None, padding_right=None,
                 padding_top=None, padding_bottom=None,
                 window_too_small=None,
                 width=None, height=None, z_index=None,
                 modal=False, key_bindings=None, style=''):
        self.table = table
        self.row = row
        self.merge = merge

        if padding is None:
            padding = D(preferred=0)

        def get(value):
            if value is None:
                value = padding
            return to_dimension(value)

        self.padding_left = get(padding_left)
        self.padding_right = get(padding_right)
        self.padding_top = get(padding_top)
        self.padding_bottom = get(padding_bottom)

        children = []
        children.append(Window(width=self.padding_left, char=char))
        if cell:
            children.append(cell)
        children.append(Window(width=self.padding_right, char=char))

        children = [
            Window(height=self.padding_top, char=char),
            VSplit(children),
            Window(height=self.padding_bottom, char=char),
        ]

        super().__init__(
            children=children,
            window_too_small=window_too_small,
            width=width,
            height=height,
            z_index=z_index,
            modal=modal,
            key_bindings=key_bindings,
            style=style)


def demo():
    txt1 = "Lorem ipsum dolor sit amet, consectetur adipiscing"
    txt2 = "Praesent eu ultrices massa. Cras et dui bibendum"
    txt3 = "Proin in varius purus. <b>Aliquam nec nulla</b>"

    sht1 = "Hello World"
    sht2 = "Buzz"
    sht3 = "The quick brown fox jumps over the lazy dog."

    kb = KeyBindings()
    @kb.add('c-c')
    def _(event):
        " Abort when Control-C has been pressed. "
        event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

    @kb.add("tab")
    def _(event):
        event.app.layout.focus_next()

    @kb.add("s-tab")
    def _(event):
        event.app.layout.focus_previous()

    def abort():
        " Abort when Control-C has been pressed. "
        get_app().exit(result=True)

    buffers = [TextArea(txt1, style="fg:ansigreen"), TextArea(txt2), TextArea(txt3)]

    def save():
        x=list(get_app().layout.find_all_controls())
        for y in buffers:
            try:
                print(y.text)
                print("--")
            except: pass

    table_1 = [
        [Label('field1', style="fg:ansicyan"), buffers[0]],
        [Label('field2'), buffers[1]],
        [Label('field3'), buffers[2]]
    ]

    table_2 = [
        [Button('Save', handler=save), Button('Abort',handler=abort)],
    ]

    
    # table = TextArea(txt2)

    layout = Layout(
        HSplit([
            Table(
                table=table_1,
                column_widths=[D(10, 30), D(30, 80)],
                borders=RoundedBorder),
            HorizontalLine(),
            Table(
                table=table_2,
                column_widths=[D(10, 30), D(30, 80)],
                borders=RoundedBorder)
            ]
        ),
    )
    return Application(layout, key_bindings=kb, full_screen=True)


if __name__ == '__main__':
    demo().run()
