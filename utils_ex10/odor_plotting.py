import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from ipywidgets import Button, IntSlider, Layout, Output, VBox

from utils_ex10.odor import APC_circuit

log_weights = False


def odor_model():
    output_plots = [Output() for _ in range(6)]
    output_descriptions = [HTML('') for _ in range(6)]
    loading_output = Output()  # Widget to show loading message
    output_text = widgets.Output()

    def update_plot(learning_rule: str):
        """
        Update the plot (triggered by the button)
        Args:
            learning_rule: Learning rule to be used
        """

        def plot_message(text: str):
            plt.figure()
            plt.text(0.5, 0.5, 'No weights to display', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.show()

        n_PN = PN_slider.value
        # n_PV = PV_slider.value
        # n_SST = SST_slider.value
        # n_VIP = VIP_slider.value

        def no_training():
            global log_weights
            mynn = APC_circuit(
                n_PN=n_PN,
                # n_PV=n_PV,
                # n_SST=n_SST,
                # n_VIP=n_VIP,
            )
            mynn.simulate()
            log_weights = False
            return mynn

        def hebbian_training():
            global log_weights
            mynn = APC_circuit(
                n_PN=n_PN,
                # n_PV=n_PV,
                # n_SST=n_SST,
                # n_VIP=n_VIP,
            )
            mynn.simulate_hebbian()
            log_weights = True
            return mynn

        def oja_training():
            global log_weights
            mynn = APC_circuit(
                n_PN=n_PN,
                # n_PV=n_PV,
                # n_SST=n_SST,
                # n_VIP=n_VIP,
            )
            mynn.simulate_oja()
            log_weights = False
            return mynn

        # Clear all plots and show loading message
        for op in output_plots:
            op.clear_output(wait=True)
        with loading_output:
            loading_output.clear_output(wait=True)
            print("Training in progress, please wait...")

        # Disable button while training is running
        plot_button.disabled = True

        try:
            # Select the appropriate training function
            if learning_rule == 'No Learning':
                mynn = no_training()
            elif learning_rule == 'Hebbian':
                mynn = hebbian_training()
            else:
                mynn = oja_training()

            mynn.getEnsembles()

            with output_plots[0]:
                mynn.plotConnectivity()

            with output_plots[1]:
                mynn.plotActivityPub()

            with output_plots[2]:
                mynn.plotEnsembleStrength()

            with output_plots[3]:
                if learning_rule == 'No Learning':
                    plot_message('No weights to display (no learning)')
                else:
                    mynn.plotWeightDistribution()

            with output_plots[4]:
                if learning_rule == 'No Learning':
                    plot_message('No network to display (no learning)')
                else:
                    mynn.plotNetwork(log_weights=log_weights)

            with output_plots[5]:
                if learning_rule == 'No Learning':
                    plot_message('No ensemble to display (no learning)')
                else:
                    mynn.plotEnsembleWeights(log_weights=log_weights)
        finally:
            # Re-enable the button when training is complete
            plot_button.disabled = False

            # Clear the loading message and show the plots
            loading_output.clear_output()

    style = {'description_width': 'initial'}

    learning_rule = widgets.Dropdown(
        options=['No Learning', 'Hebbian', 'Oja'],
        value='No Learning',
        description='Learning Rule:',
        style=style,
        layout=Layout(width='500px')
    )

    PN_slider = IntSlider(value=50, min=2, max=50, step=1, description='Number of PNs:', style=style)
    # PV_slider = IntSlider(value=4, min=1, max=10, step=1, description='Number of PVs:', style=style)
    # SST_slider = IntSlider(value=8, min=1, max=10, step=1, description='Number of SSTs:', style=style)
    # VIP_slider = IntSlider(value=1, min=1, max=10, step=1, description='Number of VIPs:', style=style)

    output_descriptions[0] = HTML(
        "<h4>Explanation</h4>"
        "Connectivity matrix shows the synaptic connections between the pyramid neurons. The rows represent the "
        "presynaptic neurons and the columns represent the postsynaptic neurons. As you can see, we have a recurrent "
        "connectivity between PNs of about 40% (as specified in the code) and no PN projects back to itself (empty "
        "diagonal in connectivity matrix).")

    output_descriptions[1] = HTML("""<h4>Explanation</h4><p>This is the activity of the different neurons at the beginning and at the end 
        of the simulation. 
        Each color corresponds to the activity of a single neuron. The activity pattern depends on the training progress of the 
        network and the input pattern. Note that the odor input is identical at time t=0.0 and t=39.6, and at t=0.6 and 
        t=39.9.</p>
        <br> 
        <p>
        <b>Assignment 14</b>
        <br>
        What differences do you observe between the different learning rules?
        </p>
        """)

    output_descriptions[2] = HTML("""<h4>Explanation</h4><p>Here, we extract the neuronal ensembles that respond to each of the two odors 
        and check their activity before and after training.</p>
        <br> 
        <p>
        <b>Assignment 15</b>
        <br>
        What differences do you observe between the different learning rules?
        </p>
        """)

    output_descriptions[3] = HTML("""<h4>Explanation</h4><p>These histograms display the weights of the PN-PN connections.</p>
        <br> 
        <p>
        <b>Assignment 16</b>
        <br>
        What differences do you observe between the different learning rules?
        </p>
        """)

    output_descriptions[4] = HTML("""<h4>Explanation</h4><p>Here we show the PN-neurons at the end of the simulation. Each vertice represents a neuron and the edges represent the strength of the weights between neurons. The stronger the weights are between two neurons, the closer they are located in space and so the formation of specific odor ensembels can clearly be seen. Of course, in reality, neurons that become part of such an ensemble, stay distributed throughout the APC.</p>
    <h5>Neuron Types</h5>
    <ul>
      <li><strong>Teacher Neurons</strong>: These neurons respond specifically to the odor linked with the Teacher signal, representing a learned or targeted response.</li>
      <li><strong>Control Neurons</strong>: These neurons respond to an odor that is <em>not</em> linked to the Teacher signal, acting as a baseline comparison without targeted learning.</li>
      <li><strong>Between Neurons</strong>: These neurons are not part of a neuronal ensemble.</li>
    </ul>
    <br>
    <p>
    <b>Assignment 17</b>
    <br>
    Odor intensity is encoded by the same neuronal ensemble that encodes odor identity. Since the membership code is invariant to concentration, how else could it be achieved?
    </p>
    """)

    output_descriptions[5] = HTML("""<h4>Explanation</h4><p>In this plot, you will visualize the <strong>evolution of mean synaptic weights</strong> in a neural network model during odor learning. Specifically, you will differentiate between:</p>
<ul>
  <li><strong>"Within" weights</strong>: The mean weights connecting neurons within the same ensemble (Teacher or Control).</li>
  <li><strong>"Between" weights</strong>: The weights between neurons that do not belong to the same ensemble (e.g., connections between Teacher and Control neurons).</li>
</ul>
<h5>Neuron Types</h5>
<ul>
  <li><strong>Teacher Neurons</strong>: These neurons respond specifically to the odor linked with the Teacher signal, representing a learned or targeted response.</li>
  <li><strong>Control Neurons</strong>: These neurons respond to an odor that is <em>not</em> linked to the Teacher signal, acting as a baseline comparison without targeted learning.</li>
</ul>
<br>
<p>
<b>Assignment 18</b>
<br>
Analyze the differences in the weight evolution for the Teacher and Control ensembles for the Oja learning rule. Why might the Control odor be learned faster than the Teacher odor?
</p>
""")

    plot_button = Button(description="Update Plot", button_style='success')

    def button_plot_on_click_action(_=None):
        update_plot(learning_rule.value)

    def update_text(change):
        if change['new'] is not None:
            with output_text:
                output_text.clear_output(wait=True)
                display(output_descriptions[change['new']])

    plot_button.on_click(button_plot_on_click_action)

    tab = widgets.Tab()
    tab.children = [VBox(
        [learning_rule, PN_slider, plot_button, loading_output, output_plots[i]]) for
                    i in range(6)]
    tab.titles = ('Connectivity', 'Activity', 'Ensemble Strength', 'Weight Distribution', 'Network', 'Ensemble Weights')

    tab.observe(update_text, names='selected_index')

    update_text({'new': 0})
    display(tab, output_text)
    update_plot(learning_rule.value)
