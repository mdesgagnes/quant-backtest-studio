// The parts of ECharts this app uses, imported by name so the build drops
// the rest. Loaded on demand from core.tsx. Add a chart type here when a
// screen first needs it.
import * as echarts from "echarts/core";
import { BarChart, LineChart } from "echarts/charts";
import {
  AxisPointerComponent, DataZoomInsideComponent, GridComponent, MarkLineComponent, TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  LineChart, BarChart, GridComponent, TooltipComponent, AxisPointerComponent,
  MarkLineComponent, DataZoomInsideComponent, CanvasRenderer,
]);

export default echarts;
